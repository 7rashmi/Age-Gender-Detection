import cv2
import os
os.chdir('C:/Users/Dheeraj/OneDrive/Desktop/project2sem/models')

def detect_age_gender(image_path, face_cascade, age_net, gender_net):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face_roi = image[y:y + h, x:x + w]

        # Preprocess face ROI for age and gender prediction
        blob = cv2.dnn.blobFromImage(
            face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False, crop=False
        )

        # Perform age and gender prediction
        age_gender_net.setInput(blob)
        detections = age_gender_net.forward()

        # Get predicted age and gender
        age = int(detections[0][0][0][0] * 100)
        gender = "Male" if detections[0][0][0][1] > 0.5 else "Female"

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(
            image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Display the result
    cv2.imshow("Age and Gender Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to Haar cascade XML file for face detection
face_cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml"



# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load Caffe model for age and gender prediction
faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'
# Path to the input image
image_path = "path/to/input/image.jpg"

# Perform age and gender detection
detect_gender(image_path, face_cascade, gender_net)
