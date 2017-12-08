import cv2
import numpy as np
import time
import imageio

# cam = cv2.VideoCapture(0)
# time.sleep(2)
reader = imageio.get_reader('<video0>')

while True:
    # import pdb
    # pdb.set_trace()
    # ret,frame = cam.read()
    frame = reader.get_next_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()