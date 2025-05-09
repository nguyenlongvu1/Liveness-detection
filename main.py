import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX







def extract_sarima_features(ear_series):
    try:
        model = SARIMAX(ear_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 10))
        model_fit = model.fit(disp=False)
        residuals = model_fit.resid

        aic = model_fit.aic
        bic = model_fit.bic
        resid_var = np.var(residuals)
        resid_mean = np.mean(residuals)
        resid_std = np.std(residuals)

        return [aic, bic, resid_var, resid_mean, resid_std]
    except:
        return [np.nan] * 5

features = []
for sequence in X:
    features.append(extract_sarima_features(sequence))
    
df_features = pd.DataFrame(features, columns=["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"])
df_features["Label"] = y

selected_features = ["ResidualMean", "AIC", "ResidualStd", "ResidualVar", "BIC"] #select important feature
X = df_features[selected_features]
X.head()

y = pd.Series(y)
X = X.dropna().reset_index(drop=True)
y = y.iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC()

clf.fit(X_train, y_train)

""""
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
"""


print(classification_report(y_test, y_pred))








































"""from facial_landmark import FacialLandmark
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


