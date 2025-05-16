import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import numpy as np

def log_visualize(logs):
    
    frame_ids = [entry.num_frame for entry in logs]
    ear_values = [entry.ear for entry in logs]

    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, ear_values, marker='o', linestyle='-', color='b', label="EAR")

    plt.xlabel("Frame ID")
    plt.ylabel("Eye Aspect Ratio (EAR)")
    plt.title("Eye Aspect Ratio Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def frame_visualize(frame, landmarks):
    colors = {
        "left_eye_landmarks": (0, 255, 0),  # Green
        "right_eye_landmarks": (255, 0, 0),  # Blue
        "left_iris_landmarks": (0, 0, 255),  # Red
        "right_iris_landmarks": (255, 255, 0)  # Yellow
    }

    for part, points in landmarks.items():
        if part in colors:
            for landmark in points:
                cv2.circle(frame, (landmark[0], landmark[1]), 2, colors[part], -1)
            if points:
                cv2.putText(frame, part, (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, colors[part], 1)

    cv2.imshow("Facial Landmarks", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
''' 
def plot_tsne(X_test, y_pred):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_test)

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("hsv", len(np.unique(y_pred)))

    for i, label in enumerate(np.unique(y_pred)):
        idx = (y_pred == label)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Pred: {label}", alpha=0.7)

    plt.legend()
    plt.title("t-SNE visualization of predictions")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
    
'''
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    

def plot_correlation_matrix(df, features, title="Feature Correlation Matrix"):
    plt.figure(figsize=(8, 6))
    corr_matrix = df[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()
