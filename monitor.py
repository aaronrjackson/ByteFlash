import random
import cv2
import time 
import numpy as np 
import tkinter as tk # for obtaining monitor resolution
import colorsys
import os

# --- constants ---
PERIOD_MS = 256         # ms per clock cycle (one byte)
CALIBRATE_MS = 5000     # all blocks on, lets the receiver find them
LEAD_IN_MS = 2000       # black screen between calibration and data
LEAD_OUT_MS = 3000      # black screen after the last byte
BLOCK_VALUE = 0.78      # data block brightness 0-1; below 1.0 to stop the
                        # camera sensor clipping, which destroys hue info
BLOCK_SCALE = 0.09      # block size as a fraction of the screen's short edge
BLOCK_GAP = 10          # extra clearance required between blocks, px
FID_SCALE = 1.6         # corner fiducial size, relative to a block
FID_INSET = 0.02        # fiducial distance from the screen edge, fraction
IMG_PATH = "img/cobblestone.png"
SEND_SIZE = (16, 16)    # (width, height) the image is downscaled to
WINDOW = "Monitor"
# -----------------

def hue_bgr(h_opencv, v=BLOCK_VALUE):
    r, g, b = colorsys.hsv_to_rgb(h_opencv / 180.0, 1.0, v)
    return (int(b * 255), int(g * 255), int(r * 255))

# keep index 0 pure white, the rest are hues at equal spacing apart from each other
BLOCK_COLORS = [(255, 255, 255)] + [hue_bgr(int(i * 180 / 8)) for i in range(8)]

class block: 
    def __init__(self, xCoord, yCoord, bitNum, size, img, b, g, r):
        self.x = xCoord
        self.y = yCoord
        self.bitNum = bitNum
        self.size = size 
        self.img = img   
        self.b = b
        self.g = g
        self.r = r
        self.thickness = -1 # This makes a filled rectangle

        self.top_left = (int(self.x - self.size/2), int(self.y - self.size/2))
        self.btm_right = (int(self.x + self.size/2), int(self.y + self.size/2))

    def on(self):
        cv2.rectangle(self.img, self.top_left, self.btm_right,
                      BLOCK_COLORS[self.bitNum], self.thickness)

    def off(self):
        cv2.rectangle(self.img, self.top_left, self.btm_right,
                      (self.b, self.g, self.r), self.thickness)

class monitor: # Removed 'def'
    def __init__(self, img, imageToSend):
        # Numpy arrays use .shape, not cv2.CAP_PROP_FRAME_WIDTH
        self.height, self.width = img.shape[:2]

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Characteristics
        self.blockSize = int(min(self.height, self.width) * BLOCK_SCALE)
        self.drawIMG = img
        self.sendIMG = imageToSend

        self.period = int(PERIOD_MS) # ms

        # Get img color
        BGR = cv2.mean(self.drawIMG)[:3]
        self.b, self.g, self.r = [int(x) for x in BGR]

        # four always-on white corner fiducials. the receiver fits a
        # homography to these every frame, so it never picks anything
        # up in the background as a block by accident.
        self.fidSize = int(self.blockSize * FID_SCALE)
        inset = int(min(self.height, self.width) * FID_INSET) + self.fidSize // 2
        self.fidCenters = [(inset, inset),
                           (self.width - inset, inset),
                           (self.width - inset, self.height - inset),
                           (inset, self.height - inset)]
        self.drawFiducials()

        # every block must sit wholly inside the fiducial quad, so the
        # receiver can reject anything outside it.
        edge = inset + self.fidSize // 2 + BLOCK_GAP + self.blockSize // 2
        self.minX, self.maxX = edge, self.width - edge
        self.minY, self.maxY = edge, self.height - edge
        
        # Generate nine randomly placed blocks, one being the clock
        self.blockList = [] 
        while True:
            randX = random.randint(self.minX, self.maxX)
            randY = random.randint(self.minY, self.maxY)

            # Make sure no blocks overlap
            overlapping = False
            for b in self.blockList: 
                if abs(randX - b.x) <= (self.blockSize + BLOCK_GAP) and abs(randY - b.y) <= (self.blockSize + BLOCK_GAP):
                    overlapping = True
                    break
            
            if overlapping:
                continue

            if len(self.blockList) == 0:
                self.clk = block(randX, randY, 0, self.blockSize, self.drawIMG, self.b, self.g, self.r)
                self.blockList.append(self.clk)
            else:
                self.blockList.append(block(randX, randY, len(self.blockList), self.blockSize, self.drawIMG, self.b, self.g, self.r))
            
            # block list full
            if len(self.blockList) == 9:
                break

    def drawFiducials(self):
        h = self.fidSize // 2
        for cx, cy in self.fidCenters:
            cv2.rectangle(self.drawIMG, (cx - h, cy - h), (cx + h, cy + h),
                          (255, 255, 255), -1)

    def allOn(self):
        for b in self.blockList:
            b.on()

    def allOff(self):
        for b in self.blockList:
            b.off()

    def hold(self, ms):
        end = time.time() + ms / 1000.0
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                return False
            if cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF == ord('q'):
                return True
        
    def run(self):
        # Show receiver where every block is to calibrate
        self.allOn()
        cv2.imshow(WINDOW, self.drawIMG)
        cv2.waitKey(1)
        if self.hold(CALIBRATE_MS): return

        self.allOff()
        cv2.imshow(WINDOW, self.drawIMG)
        cv2.waitKey(1)
        if self.hold(LEAD_IN_MS): return

        # loop through image bytes
        loopHeight, loopWidth = self.sendIMG.shape

        ground_truth_dir = "ground_truth"
        image_name = os.path.splitext(os.path.basename(IMG_PATH))[0]
        ground_truth_file = open(
        os.path.join(ground_truth_dir, f"{image_name}_ground_truth_file.txt"), "w"
)

        for y in range(loopHeight):
            for x in range(loopWidth):
                byte = self.sendIMG[y, x] 
                byte_string = f"{byte:08b}"
                print(byte_string)

                # Write the same byte string to the text file
                ground_truth_file.write(byte_string + "\n")

                for bit_num in range(len(byte_string)+1):
                    if bit_num == 0:
                        self.blockList[bit_num].on()
                    elif byte_string[bit_num-1] == '1': 
                        self.blockList[bit_num].on()
                    else:
                        self.blockList[bit_num].off()
                
                cv2.imshow(WINDOW, self.drawIMG)
                cv2.waitKey(1)
                
                # Cycle clock
                if self.hold(self.period/2): return
                self.blockList[0].off()
                cv2.imshow(WINDOW, self.drawIMG)
                cv2.waitKey(1)
                if self.hold(self.period/2): return

        # close ground truth file
        ground_truth_file.close()

        self.allOff()
        cv2.imshow(WINDOW, self.drawIMG)
        cv2.waitKey(1)
        self.hold(LEAD_OUT_MS)
            
            


if __name__ == "__main__":
    _r = tk.Tk(); _r.withdraw()
    SW, SH = _r.winfo_screenwidth(), _r.winfo_screenheight()
    _r.destroy()

    img = np.zeros((SH, SW, 3), dtype=np.uint8)
    imgToSend = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    # Downscale the image, might look a little weird but makes the monitor more active
    resized_img = cv2.resize(imgToSend, SEND_SIZE, interpolation=cv2.INTER_AREA)

    myMonitor = monitor(img, resized_img)
    myMonitor.run()
    cv2.destroyAllWindows()