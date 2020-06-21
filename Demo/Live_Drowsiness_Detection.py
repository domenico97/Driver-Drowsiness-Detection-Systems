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
            if isSum:
                timer += 1
            elif isSum is False:
                timer -= 1
            time.sleep(1)
            #print(isSum,n)

    def terminate(self):
        self._running = False

    def increment(self, value):
        global isSum
        isSum = value

    def reset(self):
        global isSum,timer
        isSum = None
        timer = 0


timer_thread = TimerTask()
t = threading.Thread(target = timer_thread.run)
t.start()

mixer.init()
sound = mixer.Sound('Alarms/alarm.wav')

face = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('Haarcascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('Haarcascades/haarcascade_righteye_2splits.xml')


lbl = ['Close', 'Open']

model = load_model('Models/firstModel.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
start_time = 0
marginThickness = 2
rpred = [0]
lpred = [0]
lpredindex = None
rpredindex = None
IMG_HEIGHT = 52
IMG_WIDTH = 52

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    lpredindex, rpredindex = None,None
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1)
    left_eye = leye.detectMultiScale(frame, minNeighbors=7)
    right_eye = reye.detectMultiScale(frame, minNeighbors=7)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]

        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (IMG_WIDTH, IMG_HEIGHT))
        #cv2.imshow('left', l_eye)

        l_eye = l_eye/255
        l_eye = l_eye.reshape(IMG_WIDTH, IMG_HEIGHT, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        #l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        lpredindex = np.argmax(lpred)
        #lbl = lbl[lpredindex]
        lpredconf = np.max(lpred)
        cv2.putText(frame, str(lpredindex)+":"+str("{:.2f}".format(lpredconf)), (x, y - 20), font, 1, (255, 255, 255), 1)
        #print("left", lpred, lpredindex, lpredconf)
        break

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]

        count = count+1

        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (IMG_WIDTH, IMG_HEIGHT))
        #cv2.imshow('right', r_eye)

        r_eye = r_eye/255
        r_eye = r_eye.reshape(IMG_WIDTH, IMG_HEIGHT, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        rpredindex = np.argmax(rpred)
        #lbl = lbl[rpredindex]
        rpredconf = np.max(rpred)
        cv2.putText(frame,str(rpredindex)+":"+str("{:.2f}".format(rpredconf)),(x,y-20),font, 1, (255, 255, 255), 1)
       
        #print("right",rpred,rpredindex,rpredconf)
        break


    #print(lpredindex,rpredindex)
    if lpredindex == 0 and rpredindex == 0:
        score = score+1
        timer_thread.increment(True)
        #start_time = time.time()
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score-1
        #start_time = 0
        timer_thread.increment(False)
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0 or timer < 0:
        score = 0
        timer_thread.reset()

    cv2.putText(frame, 'Score:'+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Timer:' + str(timer), (210, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if timer > 5:
        #person is feeling sleepy so we beep the alarm
        #cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if marginThickness < 16:
            marginThickness = marginThickness + 2
        else:
            marginThickness = marginThickness - 2
            if marginThickness < 2:
                marginThickness = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), marginThickness)

    cv2.imshow('Driver Drowsiness Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        timer_thread.terminate()
        break
cap.release()
cv2.destroyAllWindows()
