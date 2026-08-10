import random
import cv2
import time 
import numpy as np 

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
        match self.bitNum:
            # clock
            case 0:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (0, 255, 0), self.thickness) # Green
            # byte
            case 1:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (255, 0, 0), self.thickness) # Blue
            case 2: 
                cv2.rectangle(self.img, self.top_left, self.btm_right, (0, 0, 255), self.thickness) # Red
            case 3: 
                cv2.rectangle(self.img, self.top_left, self.btm_right, (0, 255, 255), self.thickness) # Yellow
            case 4:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (255, 0, 255), self.thickness) # Magenta
            case 5:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (255, 255, 0), self.thickness) # Cyan
            case 6:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (255, 255, 255), self.thickness) # White
            case 7:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (128, 128, 128), self.thickness) # Gray
            case 8:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (0, 165, 255), self.thickness) # Orange

    def off(self):
        # All cases resolve to the same color in off state, simplified match statement
        match self.bitNum:
            case _:
                cv2.rectangle(self.img, self.top_left, self.btm_right, (self.b, self.g, self.r), self.thickness) 

class monitor: # Removed 'def'
    def __init__(self, img, imageToSend):
        # Numpy arrays use .shape, not cv2.CAP_PROP_FRAME_WIDTH
        self.height, self.width = img.shape[:2]

        # Characteristics
        self.blockSize = 30
        self.drawIMG = img
        self.sendIMG = imageToSend

        self.period = int(30) # ms

        # Get img color
        BGR = cv2.mean(self.drawIMG)[:3]
        self.b, self.g, self.r = [int(x) for x in BGR]
        
        # Generate nine randomly placed blocks, one being the clock
        self.blockList = [] 
        while True:
            randX = random.randint(int(self.blockSize/2), int(self.width - self.blockSize))
            randY = random.randint(int(self.blockSize/2), int(self.height - self.blockSize))

            # Make sure no blocks overlap
            overlapping = False
            for b in self.blockList: 
                if abs(randX - b.x) <= (self.blockSize+10) and abs(randY - b.y) <= (self.blockSize + 10):
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

    def allOn(self):
        for b in self.blockList:
            b.on()

    def allOff(self):
        for b in self.blockList:
            b.off()
        
    def run(self):
        # Show receiver where every block is to calibrate
        self.allOn()

        cv2.imshow("Monitor", self.drawIMG)
        cv2.waitKey(1) 
        if cv2.waitKey(5000) & 0xFF == ord('q'): return
        
        self.allOff()
        cv2.imshow("Monitor", self.drawIMG)
        cv2.waitKey(1)
        if cv2.waitKey(5000) & 0xFF == ord('q'): return

        # loop through image bytes
        loopHeight, loopWidth = self.sendIMG.shape

        for y in range(loopHeight):
            for x in range(loopWidth):
                byte = self.sendIMG[y, x] 
                byte_string = f"{byte:08b}"
                print(byte_string)
                for bit_num in range(len(byte_string)+1):
                    if bit_num == 0:
                        self.blockList[bit_num].on()
                    elif byte_string[bit_num-1] == '1': 
                        self.blockList[bit_num].on()
                    else:
                        self.blockList[bit_num].off()
                
                cv2.imshow("Monitor", self.drawIMG)
                cv2.waitKey(1)
                
                # Cycle clock
                if cv2.waitKey(int(self.period/2)) & 0xFF == ord('q'): return
                self.blockList[0].off()
                cv2.imshow("Monitor", self.drawIMG)
                cv2.waitKey(1)
                if cv2.waitKey(int(self.period/2)) & 0xFF == ord('q'): return
            
            


if __name__ == "__main__":
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    imgToSend = cv2.imread("img/waterBucket.png", cv2.IMREAD_GRAYSCALE)

    # Downscale the image, might look a little weird but makes the monitor more active
    resized_img = cv2.resize(imgToSend, (16,16), interpolation=cv2.INTER_AREA)

    myMonitor = monitor(img, resized_img)
    myMonitor.run()
    cv2.destroyAllWindows()