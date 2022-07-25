import cv2


def detect_faces(frame, classifier):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = classifier.detectMultiScale(frame, 1.1, 4)
    return faces


cap = cv2.VideoCapture(0)

# load classifier
classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

while cap.isOpened():
    _, frame = cap.read()

    faces = detect_faces(frame, classifier)
    print(faces)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (23, 44, 199), 3)
    #     cv2.putText(frame, 'Face', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 2, (23, 44, 199), 2)

    cv2.imshow('Detect Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()