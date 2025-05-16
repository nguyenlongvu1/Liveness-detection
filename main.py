from sarima_features import extract_all_features
from visualize import plot_correlation_matrix
import pandas as pd

# Bước 1: Trích xuất đặc trưng từ video
df_features = extract_all_features(video_dir="C:\\Users\\Admin\\Downloads\\code\\datasets\\train\\train\\videos",
                                   label_path="C:\\Users\\Admin\\Downloads\\code\\datasets\\train\\train\\label.csv",
                                   max_videos=600)

selected_features = ["ResidualMean", "AIC", "ResidualStd", "ResidualVar", "BIC"]
plot_correlation_matrix(df_features, selected_features)

# Lưu feature để models.py dùng
df_features.to_csv("data/features/features.csv", index=False)


































































































































'''

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
from sklearn.manifold import TSNE
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



def select_best_sarima_params(ear_seq, s=90):
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    for p in range(2):
        for d in range(2):
            for q in range(2):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            try:
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, s)
                                model = SARIMAX(ear_seq, order=order, seasonal_order=seasonal_order,
                                                enforce_stationarity=False, enforce_invertibility=False)
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                            except:
                                continue
    return best_order, best_seasonal_order

def extract_sarima_features(ear_seq):
    try:
        order, seasonal_order = select_best_sarima_params(ear_seq, s=10)
        model = SARIMAX(ear_seq, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        residuals = model_fit.resid
        return [
            model_fit.aic,
            model_fit.bic,
            np.var(residuals),
            np.mean(residuals),
            np.std(residuals)
        ]
    except Exception as e:
        print(f"[!] SARIMA failed on sequence: {e}")
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

# Áp dụng t-SNE lên dữ liệu đã scale
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_test)

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



# Vẽ scatter plot phân biệt theo nhãn dự đoán
plt.figure(figsize=(8, 6))
palette = sns.color_palette("hsv", len(np.unique(y_pred)))

for i, label in enumerate(np.unique(y_pred)):
    idx = (y_pred == label)
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Predicted: {label}", alpha=0.7, s=60)

plt.title("t-SNE visualization of SVM predictions")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()



































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
'''


