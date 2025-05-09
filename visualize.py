import cv2
import matplotlib.pyplot as plt

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
