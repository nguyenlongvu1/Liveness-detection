from utils import EAR
import cv2
from facial_landmark import FacialLandmark
from log_time import log
from visualize import log_visualize, frame_visualize
import math
import statistics


detector = FacialLandmark(refine_landmarks=True)
logs = {}
threshold = {}
MAX_INT = 1000000000000000000000000000000000000000000

def thres_calc(logs):
    mean = sum([log.ear for log in logs])/ (len(logs))
    
    res = math.sqrt(sum([(log.ear - mean)**2 for log in logs])/(len(logs) - 1))

    return res


def EAR_calc(landmarks):
    left_EAR = EAR(landmarks["left_eye_landmarks"])
    right_EAR = EAR(landmarks['right_eye_landmarks'])
    return (left_EAR + right_EAR) / 2

def group_frame(lst, diff):
    if not lst:
        return []
    groups = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i].ear - lst[i - 1].ear < diff:
            current_group.append(lst[i])
        else:
            groups.append(min(current_group, key=lambda x: x.ear))  # Keep only the smallest `ear`
            current_group = [lst[i]]

    groups.append(min(current_group, key=lambda x: x.ear))  # Add the last group
    return groups


def analyse_blink(close_eyes):
    blinks = (group_frame(close_eyes, 2))

    if(len(blinks) < 1): return True

    return False

def logging(people_id, frame_id, frame_path):
    frame = cv2.imread(frame_path)
    frame, landmarks = detector.findEyeLandmark(frame)

    eye_EAR = EAR_calc(landmarks)
    #frame_visualize(frame, landmarks)
    
    if people_id not in logs: 
        logs[people_id] = []

    logs[people_id].append(log(frame_id, eye_EAR))
    # print(logs.keys())
    
    if people_id not in EAR_score:
        EAR_score[people_id] = []
    
    EAR_score[people_id].append(eye_EAR)

    if len(logs[people_id]) > 30: #if t-t> threshold
        print(people_id)
        log_visualize(logs[people_id])
        if people_id not in threshold: threshold[people_id] = 0
        threshold[people_id] = thres_calc(logs[people_id])

        close_eye_frame = []
        
        for log_fr in logs[people_id]:
            print(log_fr.ear, threshold[people_id])
            if((log_fr.ear - threshold[people_id])) < 0:
                close_eye_frame.append(log_fr)
        
        print(analyse_blink(close_eye_frame))
    return EAR_score


                









