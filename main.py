import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm
from eye_tracking import logging
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

video_dir = "C:\\Users\\Admin\\Downloads\\code\\datasets\\train\\train\\videos"
label_path = "C:\\Users\\Admin\\Downloads\\code\\datasets\\train\\train\\label.csv"

# Đọc nhãn
df_label = pd.read_csv(label_path)
labels = dict(zip(df_label['people_id'].astype(str), df_label['liveness_score']))

# Biến lưu kết quả
ear_series = []
y = []

max_videos = 100
video_count = 0

video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')], key=lambda x: int(x.replace('.mp4', '')))

# Duyệt từng video
for video_name in tqdm(video_files):
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    people_id = video_name.replace(".mp4", "")
    ear_scores = []
    video_valid = True
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            ear = logging(temp_path)
            ear_scores.append(ear)
        except:
            print(f"[!] Error in {video_name} at frame {frame_id}, skipping this video.")
            video_valid = False
            break

        frame_id += 1

    cap.release()

    # Nếu hợp lệ, lưu lại
    if video_valid and len(ear_scores) > 0 and people_id in labels:
        ear_series.append(ear_scores)
        y.append(labels[people_id])
        video_count += 1

    if video_count >= max_videos:
        break


# Lưu X, y nếu cần
# with open("ear_sequences.pkl", "wb") as f:
#     pickle.dump((X, y), f)


def extract_sarima_features(ear_seq):
    try:
        model = SARIMAX(ear_seq, order=(1, 0, 1), seasonal_order=(1, 0, 1, 10))
        model_fit = model.fit(disp=False)
        residuals = model_fit.resid
        return [
            model_fit.aic,
            model_fit.bic,
            np.var(residuals),
            np.mean(residuals),
            np.std(residuals)
        ]
    except:
        return [np.nan] * 5

features = []
for sequence in ear_series:
    features.append(extract_sarima_features(sequence))
    
df_features = pd.DataFrame(features, columns=["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"])
df_features["Label"] = y

selected_features = ["ResidualMean", "AIC", "ResidualStd", "ResidualVar", "BIC"] #select important feature
X = df_features[selected_features]


y = pd.Series(y)
X = X.dropna().reset_index(drop=True)
y = y.iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)


print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()







































"""
from facial_landmark import FacialLandmark
import cv2
from utils import EAR

# Initialize Mediapipe Facial Landmark Detector
detector = FacialLandmark(refine_landmarks=True)

# Read an image from a specified file path
image_path = "C:\\Users\\Admin\\Downloads\\code\\dataset\\full_HUST_LEBW\\training\\training\\blink\\3\\10\\00066.bmp"  # Replace with your actual image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # Use FaceMesh to find facial landmarks
    image, landmarks = detector.findEyeLandmark(image)
"""


