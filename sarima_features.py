import pandas as pd
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from eye_tracking import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

def select_best_sarima_params(ear_seq, s=10):
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
        order, seasonal_order = select_best_sarima_params(ear_seq)
        model = SARIMAX(ear_seq, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        residuals = model_fit.resid
        features = [
            model_fit.aic,
            model_fit.bic,
            np.var(residuals),
            np.mean(residuals),
            np.std(residuals)
        ]
        return features, order, seasonal_order
    except Exception as e:
        print(f"[!] SARIMA failed: {e}")
        return [np.nan]*5, None, None

def extract_all_features(video_dir, label_path, max_videos=100, save_path="models/sarima_features.pkl"):
    df_label = pd.read_csv(label_path)
    labels = dict(zip(df_label['people_id'].astype(str), df_label['liveness_score']))

    video_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith('.mp4')],
        key=lambda x: int(x.replace('.mp4', ''))
    )

    features = []
    y = []
    sarima_orders = []
    sarima_seasonals = []
    video_count = 0
    error_videos = []

    for video_name in tqdm(video_files):
        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
        people_id = video_name.replace(".mp4", "")
        ear_scores = []
        valid = True

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
                valid = False
                error_videos.append(video_name)  # Ghi lại tên video lỗi
                break
        cap.release()

        if valid and len(ear_scores) > 0 and people_id in labels:
            feat, order, seasonal = extract_sarima_features(ear_scores)
            features.append(feat)
            sarima_orders.append(order)
            sarima_seasonals.append(seasonal)
            y.append(labels[people_id])
            video_count += 1

        if video_count >= max_videos:
            break

    df = pd.DataFrame(features, columns=["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"])
    df["Label"] = y
    df["SARIMA_Order"] = sarima_orders
    df["SARIMA_Seasonal"] = sarima_seasonals

    # Save to file for reuse
    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    # In ra các video bị lỗi
    if error_videos:
        print("\n[!] Các video bị lỗi khi xử lý:")
        for vid in error_videos:
            print(f" - {vid}")

    return df



