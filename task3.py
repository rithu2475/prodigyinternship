import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
# Settings
image_size = (64, 64)  # Resize all images to 64x64

data = []
labels = []

# Load cat images
for file in os.listdir("dataset/cats"):
    img = imread(os.path.join("dataset/cats", file), as_gray=True)
    img_resized = resize(img, image_size)
    data.append(img_resized.flatten())
    labels.append(0)  # 0 for cat

# Load dog images
for file in os.listdir("dataset/dogs"):
    img = imread(os.path.join("dataset/dogs", file), as_gray=True)
    img_resized = resize(img, image_size)
    data.append(img_resized.flatten())
    labels.append(1)  # 1 for dog

X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')  # You can try 'rbf' or 'poly' as well
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
