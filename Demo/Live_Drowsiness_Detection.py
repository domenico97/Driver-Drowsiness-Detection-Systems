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
        global n, isSum
        n = 0
        isSum = None
        print("TimerTask init")

    def run(self):
        print("TimerTask run")
        global n
        while self._running:
            if isSum:
                n += 1
            elif isSum is False:
                n -= 1
            time.sleep(1)
            #print(isSum,n)

    def terminate(self):
        self._running = False

    def increment(self, value):
        global isSum
        isSum = value

    def reset(self):
        global isSum,n
        isSum = None
        n = 0


timer_thread = TimerTask()
t = threading.Thread(target = timer_thread.run)
t.start()

mixer.init()
sound = mixer.Sound('Alarms/alarm.wav')

face = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('Haarcascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('Haarcascades/haarcascade_righteye_2splits.xml')


lbl = ['Close', 'Open']

model = load_model('Models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
start_time = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        if rpred[0] == 1:
            lbl = 'Open'
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye= l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if lpred[0] == 1:
            lbl = 'Open'
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
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

    if score < 0 or n < 0:
        score = 0
        timer_thread.reset()

    cv2.putText(frame, 'Score:'+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Timer:' + str(n), (210, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if thicc < 16:
            thicc = thicc+2
        else:
            thicc = thicc-2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        timer_thread.terminate()
        break
cap.release()
cv2.destroyAllWindows()
