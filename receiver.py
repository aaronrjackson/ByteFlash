import cv2
import numpy as np
import colorsys

### --- constants ---
VIDEO = "vid/test.mov"          # recording of the transmitter

CAL_SCAN_S = 10.0               # seconds from start of footage to search for the calibration window
FID_JUMP = 2.5                  # max fiducial (the corner boxes) movement between frames, in fid widths

DEBUG_VIDEO = "out/debug.mp4"   # annotated detection video, None to skip
DEBUG_SCALE = 1.0               # 1.0 keeps full resolution
DEBUG_TEXT = 1.8                # multiplier on debug font size

BLOCK_VALUE = 0.78              # MUST match same value in monitor.py.
### -----------------


# rebuild one of the transmitter's block colors from its hue
def hue_bgr(h_opencv, v=BLOCK_VALUE):
    r, g, b = colorsys.hsv_to_rgb(h_opencv / 180.0, 1.0, v)
    return (int(b * 255), int(g * 255), int(r * 255))

# the nine colors we expect to see: white clock, then eight evenly spaced hues
REF_BGR = [(255, 255, 255)] + [hue_bgr(int(i * 180 / 8)) for i in range(8)]

LABELS = ["CLK"] + [f"b{7 - i}" for i in range(8)]
LUMA = np.array([0.114, 0.587, 0.299])


# find bright, square-ish, solid regions in the frame
# returns a list of (cx, cy, w, h, area) tuples, largest first.
def find_blobs(frame):
    
    # split bright pixels from the dark background
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # erase bits of noise before looking for shapes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0.05 * frame.shape[0] * frame.shape[1]
    out = []
    for c in cnts:
        # throw out anything too small, too big, not square, or not solid
        a = cv2.contourArea(c)
        if a < 80 or a > max_area: # size check
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or abs(1 - w / h) > 0.45: # aspect ratio check
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or a / hull < 0.8: # solidity check
            continue
        out.append((x + w // 2, y + h // 2, w, h, a))
    out.sort(key=lambda b: -b[4])
    return out

# sort fiducials into top-left, top-right, bottom-right, bottom-left
def order_quad(pts):
    p = np.array(pts, dtype=np.float32)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.float32([p[np.argmin(s)], p[np.argmin(d)],
                       p[np.argmax(s)], p[np.argmax(d)]])

# get median color of a rectangular region of interest in the frame
def patch(frame, roi):
    cx, cy, w, h = roi
    hw, hh = max(2, int(w * 0.3)), max(2, int(h * 0.3))
    p = frame[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw]
    return np.median(p.reshape(-1, 3), axis=0) if p.size else np.zeros(3) # take median not mean!!

# convert a BGR color to Lab color with a/b only, dropping L
def chroma(bgr):
    px = np.uint8([[[int(np.clip(c, 0, 255)) for c in bgr]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0, 1:].astype(float)

# match the nine calibrated block colors to the nine blobs we see in this frame
def match_colors(samples):
    # score every possible block-to-blob pairing, closest colors first
    pairs = sorted((np.linalg.norm(chroma(REF_BGR[n]) - chroma(samples[i])), n, i)
                   for n in range(9) for i in range(len(samples)))

    # take the best pairings, never reusing a block or a blob
    order, un, ui = [None] * 9, set(), set()
    for _, n, i in pairs:
        if n not in un and i not in ui:
            order[n] = i
            un.add(n)
            ui.add(i)
    return order

# match the four largest blobs near the previous known fiducial positions.
def track_fiducials(blobs, prev, jump):
    got = []
    used = set()
    for px, py in prev:
        # Find the nearest unclaimed blob to where this corner was last frame
        best, bi = jump, None
        for i, (bx, by, _, _, _) in enumerate(blobs):
            if i in used:
                continue
            d = np.hypot(bx - px, by - py)
            if d < best:
                best, bi = d, i

        # any corner we cannot find means we cannot trust the whole fit
        if bi is None:
            return None
        used.add(bi)
        got.append((blobs[bi][0], blobs[bi][1]))
    return np.float32(got)


def warp_rois(cal_rois, H):
    """Push the calibrated block positions through the homography."""
    # Move every block centre to where it sits in this frame
    pts = np.float32([[[cx, cy]] for cx, cy, _, _ in cal_rois])
    moved = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    # Warp a test square to see how much bigger or smaller the screen got
    corners = np.float32([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    warped = cv2.perspectiveTransform(corners, H)[0]
    scale = np.sqrt(abs(cv2.contourArea(warped)) / 10000.0) or 1.0
    return [(int(px), int(py), max(4, int(w * scale)), max(4, int(h * scale)))
            for (px, py), (_, _, w, h) in zip(moved, cal_rois)]

# text helper function for debug outputs
def text(vis, s, org, fs, color, th):
    cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                th + 3, cv2.LINE_AA)
    cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color,
                th, cv2.LINE_AA)

# used for drawing what the detector "sees" every frame for debug outputs
def annotate(frame, rois, lit, quad):
    vis = frame.copy()
    fs = max(0.6, frame.shape[0] / 1400.0) * DEBUG_TEXT
    th = max(2, int(fs * 2))

    # yellow outline through fiducials
    if quad is not None:
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255),
                      max(2, th))

    # box and label each block, bright when we read it as on
    for n, (cx, cy, w, h) in enumerate(rois):
        tl = (cx - w // 2, cy - h // 2)
        br = (cx + w // 2, cy + h // 2)
        if lit[n]:
            cv2.rectangle(vis, tl, br, (0, 0, 0), max(3, th * 3))
            cv2.rectangle(vis, tl, br, REF_BGR[n], max(2, th * 2))
        else:
            cv2.rectangle(vis, tl, br, (70, 70, 70), max(1, th // 2))
        text(vis, f"{LABELS[n]}={int(lit[n])}", (tl[0], tl[1] - int(10 * fs)),
             fs, REF_BGR[n] if lit[n] else (140, 140, 140), th)

    # status readout in the top corner
    bits = "".join(str(int(b)) for b in lit[1:])
    lines = [
        f"clk {'HIGH' if lit[0] else 'low '}   "
        f"{'LOCKED' if quad is not None else 'NO LOCK'}",
        f"bits {bits}",
    ]
    y = int(50 * fs)
    for line in lines:
        text(vis, line, (14, y), fs, (255, 255, 255), th)
        y += int(46 * fs)
    return vis


# open the recording and report its frame rate
cap = cv2.VideoCapture(VIDEO) # video file processed for cv2
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"{VIDEO}: {fps:.1f} fps")

# calibrate across the whole calibration window timespan instead of one frame.
# averaging several seconds of samples removes most of the sensor noise that makes the
# saturated colors hard to tell apart.
cal_rois, cal_quad, cal_idx, nframes = None, None, 0, 0
acc = np.zeros((9, 3))
for i in range(int(CAL_SCAN_S * fps)):
    ok, frame = cap.read()
    if not ok:
        break

    # skip frames that do not show all four fiducials plus all nine blocks
    blobs = find_blobs(frame)
    if len(blobs) < 13:
        continue

    if cal_rois is None:
        # four largest are the corner fiducials
        # blocks are the nine largest remaining that fall strictly inside the fiducial quadrilateral
        quad = order_quad([(b[0], b[1]) for b in blobs[:4]])
        inside = [b for b in blobs[4:]
                  if cv2.pointPolygonTest(quad, (float(b[0]), float(b[1])), False) > 0]
        if len(inside) < 9:
            continue
        cal_quad = quad
        cal_rois = [(b[0], b[1], b[2], b[3]) for b in inside[:9]]
        cal_idx = i

    # add this frame's colors to the running total
    acc += np.array([patch(frame, r) for r in cal_rois])
    nframes += 1
if cal_rois is None:
    raise SystemExit("never saw 4 fiducials plus 9 blocks during calibration; check framing, focus, and that the whole screen is in view")

# average the samples, then work out which block is which color
mean_bgr = acc / nframes
order = match_colors(list(mean_bgr))
cal_rois = [cal_rois[j] for j in order]
on_luma = [float(mean_bgr[j] @ LUMA) for j in order]

# a block counts as on if it is at least half as bright as it was in calibration
thresh = [max(12.0, 0.5 * L) for L in on_luma]

# how far a corner is allowed to move between frames before we call it lost
fid_w = np.mean([b[2] for b in find_blobs(frame)[:4]]) if nframes else 60
jump = FID_JUMP * max(20.0, fid_w)

# calibration debug output
print(f"calibrated on {nframes} frames starting at frame {cal_idx}")
sep = min(np.linalg.norm(chroma(mean_bgr[order[m]]) - chroma(mean_bgr[order[n]]))
          for m in range(1, 9) for n in range(m + 1, 9))
print(f"  min chroma separation between data blocks: {sep:.1f}"
      + ("   <- too close, lower BLOCK_VALUE or monitor brightness" if sep < 12 else ""))
for n, (r, L) in enumerate(zip(cal_rois, on_luma)):
    flag = "  <- low contrast" if L < 50 else ""
    print(f"  block {n} ({LABELS[n]}): at {r[:2]}  on_luma={L:.0f}{flag}")


# track the screen and read each block's on/off state frame-by-frame
cap.set(cv2.CAP_PROP_POS_FRAMES, cal_idx + int(fps))
idx, lost = 0, 0
prev_fid = cal_quad
rois = cal_rois
writer = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1

    # refind the corners and move the block boxes to match this frame
    quad = track_fiducials(find_blobs(frame), prev_fid, jump)
    if quad is not None:
        Hm = cv2.getPerspectiveTransform(cal_quad, quad)
        rois = warp_rois(cal_rois, Hm)
        prev_fid = quad
    else:
        # corners not found, so keep using the last known positions
        lost += 1

    # read whether each block is currently lit
    lit = [patch(frame, r) @ LUMA > thresh[n] for n, r in enumerate(rois)]

    # save an annotated copy of this frame to the debug video
    if DEBUG_VIDEO:
        vis = annotate(frame, rois, lit, quad)
        if DEBUG_SCALE != 1:
            vis = cv2.resize(vis, None, fx=DEBUG_SCALE, fy=DEBUG_SCALE)
        if writer is None:
            writer = cv2.VideoWriter(DEBUG_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (vis.shape[1], vis.shape[0]))
        writer.write(vis)

# exit
cap.release()
if writer is not None:
    writer.release()
    print(f"debug video -> {DEBUG_VIDEO}")
print(f"read {idx} frames")
if lost:
    print(f"lost fiducial lock on {lost}/{idx} frames")