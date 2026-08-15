import cv2
import numpy as np

# --- constants ---
IMG_SIZE = (16, 16) #MUST be the same as SEND_SIZE in monitor.py

OUT_IMAGE = "out/decoded.png"

# -----------------


class ByteDecoder:
    #default size to IMG_SIZE
    def __init__(self, width = IMG_SIZE[0], height = IMG_SIZE[1]):
        self.width = width
        self.height = height
        
        self.area = width * height
        self.bytes = bytearray() #best data structure for 0-255 bytes

        self.prev_clock = False

    def done(self):
        if len(self.bytes) == self.area:
            return True
        return False
    
    #call once / frame, updates previous clock, if clock moves to ON then read bits into byte
    #lit is list of bits including clock
    #lit[0] = clock, lit[1 : 9] = bits b7 - b0, same as receiver.py
    def update(self, lit):
        clock_state = bool(lit[0])

        #check if clock is moving to True state, we only want to read at that instant to avoid double reads
        rising = clock_state and not self.prev_clock
        self.prev_clock = clock_state

        if not rising or self.done(): #no other updates or img finished processing
            return None
        
        byte = 0
        for bit in lit[1 : 9]:
            #format for reading bits into a byte, << is left shift op then OR with |
            byte = byte << 1 | int(bool(bit))

        self.bytes.append(byte)
        return byte
    
    #convert collection of bytes back into grayscale image array sent by monitor.py
    #bytes represent brightness val since using grayscale, otherwise would need 3 byte channels
    def to_image(self):
        #throw ValueError if trying to convert image without fully reading bits, easy for debug pinpointing
        if not self.done():
            raise ValueError(f"Trying to convert image too early, {len(self.bytes)} / {self.area} bytes read")
        
        #convert bytes into array
        flat_image = np.frombuffer(self.bytes, dtype = np.uint8)

        #reshape into IMG_SIZE dimensions
        return flat_image.reshape(self.height, self.width)
    
    #save image to path
    def save(self, path = OUT_IMAGE):
        cv2.imwrite(path, self.to_image())
        return path

        
