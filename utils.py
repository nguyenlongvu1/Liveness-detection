import numpy as np

def EAR(landmarks):
    A = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[5]))
    B = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[4]))
    C = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[3]))
    ear = (A + B) / (2.0 * C)
    if(landmarks[1][1] < landmarks[5][1]): ear = -ear
    return ear

# Algorithm 2 The Eye Tracker
# Input:
# threshold t, left eyel, right eyer
# output:
# log(blink count, time, period, elapsed time and etc)

# Main:

# Loop(frame in frames):

# LeftEAR ← Track_EAR(l)

# RightEAR ← Track_EAR(r)

# EAR ← (LeftEAR + RightEAR) / 2

# logs ← logging(frame, EAR, time_capture())

# Loop(log in logs):

# IF EAR < t:

# blink_count + = 1

# blink_time, elapsed_time AND etc ← time_analysis()

# blink_period AND etc ← period_analysis()

# Algorithm 3 A Method of Comparative Analysis
# Input:
# Eye_blink E [count, period AND etc],

# DB_Data D [count, period AND etc]

# Output:
# Fake OR Not

# Loop[i]:

# IF(E [i] <D [i]){IF(D [i]−E [i]) >= allowable range{return Fake}}

# ELSE{IF(E [i]−D [i]) >= allowable range{return Fake}}

# i++