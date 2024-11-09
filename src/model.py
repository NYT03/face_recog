import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('src/model_full.h5')  # Make sure this path is correct

# Prevent OpenCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label to an emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector (Haar Cascade classifier)
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the frame
    faces = face_casc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) for the face and resize it to 48x48 (model input size)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)  # Add batch dimension and channel dimension
        
        # Normalize the image (if required by your model)
        cropped_img = cropped_img / 255.0
        
        # Predict emotion using the model
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))  # Get the class with the highest probability
        
        # Display the predicted emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame with the face bounding box and emotion label
    cv2.imshow('Emotion Detection', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
