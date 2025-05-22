import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Settings
image_size = 64
data = []
labels = []
gesture_labels = ['palm', 'fist', 'okay', 'peace']
label_map = {gesture: i for i, gesture in enumerate(gesture_labels)}

# Load images
for gesture in gesture_labels:
    folder = f'gestures/{gesture}'
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        data.append(img)
        labels.append(label_map[gesture])

X = np.array(data).reshape(-1, image_size, image_size, 1) / 255.0
y = to_categorical(np.array(labels), num_classes=len(gesture_labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(gesture_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
model.save("hand_gesture_model.h5")
model = tf.keras.models.load_model("hand_gesture_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = cv2.cvtColor(frame[100:300, 100:300], cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi, (image_size, image_size)) / 255.0
    roi_reshaped = roi_resized.reshape(1, image_size, image_size, 1)

    pred = model.predict(roi_reshaped)
    gesture = gesture_labels[np.argmax(pred)]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f'Gesture: {gesture}', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
