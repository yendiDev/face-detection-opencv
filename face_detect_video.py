import cv2
import math


def detect_faces(frame, classifier):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = classifier.detectMultiScale(frame, 1.1, 4)
    return faces


cap = cv2.VideoCapture(0)

# load classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# face bounding box details
xface, yface, wface, hface = 0, 0, 0, 0

while cap.isOpened():
    _, frame = cap.read()

    faces = detect_faces(frame, classifier)

    if len(faces) != 1:
        # do nothing
        continue
    else:
        for (x, y, w, h) in faces:
            xface = x
            yface = y
            wface = w
            hface = h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (23, 44, 199), 3)
            cv2.putText(frame, 'Face', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 2, (23, 44, 199), 2)
    
        # calculate area of bouding box from points
        length = math.sqrt(1 + abs(yface - (yface+hface))**2)
        width = math.sqrt(abs(xface - (xface+wface))**2 + 1)
        area = length * width

        # take 10% of area
        percentage = 0.1*area

        print('Length is: ', length)
        print('Width is: ', width)
        print('Area: ', area)
        print('Percentage: ', percentage)
        print('\n\n')

    cv2.imshow('Detect Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
