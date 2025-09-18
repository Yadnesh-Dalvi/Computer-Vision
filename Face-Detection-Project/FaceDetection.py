import cv2
import os
import numpy as np

# Path where your black & white face images are stored
dataset_path = "dataset"  # e.g., dataset/user1/img1.jpg

# Create lists for training data and labels
faces = []
labels = []

label = 1  # Assign a numeric label for your face

# Load images and convert to numpy arrays
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        faces.append(np.array(img, dtype=np.uint8))
        labels.append(label)

faces = np.array(faces)
labels = np.array(labels)

# Create LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
face_recognizer.train(faces, labels)
print("Training completed!")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y + h, x:x + w]
        # Recognize face
        label_pred, confidence = face_recognizer.predict(face_roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Yadnesh Dalvi", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
