from facial_emotion_recogition import EmotionRecognition

import cv2

emotion = EmotionRecognition(device = 'gpu')

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    facial_emotion = emotion.recognize_emotion(frame, return_type = 'BGR')
    cv2.imshow('frame', frame)
    key = cv2.waitKey('q')
    if key == 'q':
        break

cam.release()
cv2.destroyAllWindows()