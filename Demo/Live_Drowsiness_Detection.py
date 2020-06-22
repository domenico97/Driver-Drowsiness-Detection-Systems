import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import threading


class TimerTask:

    def __init__(self):
        self._running = True
        global timer, isSum
        timer = 0
        isSum = None
        print("TimerTask init")

    def run(self):
        print("TimerTask run")
        global timer
        while self._running:
            #print(isSum)
            if isSum:
                #print("sum")
                timer += 1
            elif isSum is False:
                timer -= 1
            time.sleep(1)

    def terminate(self):
        self._running = False

    def increment(self, value):
        #print("TimerTask increment")
        global isSum
        isSum = value

    def reset(self):
        #print("TimerTask reset")
        global isSum, timer
        isSum = None
        timer = 0


timer_thread = TimerTask()
t = threading.Thread(target=timer_thread.run)
t.start()

mixer.init()
sound = mixer.Sound('Alarms/beep_tone.wav')

#face = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('Haarcascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('Haarcascades/haarcascade_righteye_2splits.xml')


#lbl = ['Close', 'Open']

model = load_model('Models/firstModel.h5')
#path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#count = 0
#start_time = 0
#marginThickness = 2
rpred = [0]
lpred = [0]
lpredindex = None
rpredindex = None
IMG_HEIGHT = 52
IMG_WIDTH = 52

if cap.isOpened():
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    #print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4

    print('width, height:', cap_width, cap_height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)  # float
    #print(cv2.CAP_PROP_FPS) # 5

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frames count:', frame_count)  # float
    #print(cv2.CAP_PROP_FRAME_COUNT) # 7

text = "DROWSINESS ALERT!"
# get boundary of this text
textsize = cv2.getTextSize(text, font, 2, 2)[0]

# get coords based on boundary
textX = int((cap_width - textsize[0]) / 2)
textY = int((cap_height + textsize[1]) / 2)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    lpredindex, rpredindex = None, None
    left_eye = leye.detectMultiScale(frame, minNeighbors=10)
    right_eye = reye.detectMultiScale(frame, minNeighbors=10)
    #cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        #count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (IMG_WIDTH, IMG_HEIGHT))
        #cv2.imshow('left', l_eye)
        l_eye = l_eye/255
        l_eye = l_eye.reshape(IMG_WIDTH, IMG_HEIGHT, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        lpredindex = np.argmax(lpred)
        lpredconf = np.max(lpred)
        #cv2.putText(frame, str(lpredindex)+":"+str("{:.2f}".format(lpredconf)), (x, y - 20), font, 1, (255, 255, 255), 1)

        break

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        #count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (IMG_WIDTH, IMG_HEIGHT))
        #cv2.imshow('right', r_eye)
        r_eye = r_eye/255
        r_eye = r_eye.reshape(IMG_WIDTH, IMG_HEIGHT, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        rpredindex = np.argmax(rpred)
        rpredconf = np.max(rpred)
        #cv2.putText(frame, str(rpredindex)+":"+str("{:.2f}".format(rpredconf)), (x, y-20), font, 1, (255, 255, 255), 1)

        break

    #print(lpredindex,rpredindex)
    if lpredindex == 0 and rpredindex == 0:
        timer_thread.increment(True)
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        timer_thread.increment(False)
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if timer < 0:
        timer_thread.reset()
        if mixer.get_busy():
            sound.stop()

    cv2.putText(frame, 'Timer:' + str(timer), (210, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if timer > 1:
        #person is feeling sleepy so we beep the alarm
        #cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        cv2.putText(frame, text, (textX,textY), font, 2,(0, 0, 255), 2)
        if not mixer.get_busy():
            sound.play()
        '''
        if marginThickness < 16:
            marginThickness = marginThickness + 2
        else:
            marginThickness = marginThickness - 2
            if marginThickness < 2:
                marginThickness = 2
        
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), marginThickness)
        '''
    cv2.imshow('Driver Drowsiness Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        timer_thread.terminate()
        break

cap.release()
cv2.destroyAllWindows()
