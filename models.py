from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train_svm(df):
    selected_features = ["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"]
    X = df[selected_features].dropna().reset_index(drop=True)
    y = df["Label"].iloc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC()
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))

    # Lưu model và scaler
    joblib.dump(clf, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Lưu tập test
    df_test = pd.DataFrame(X_test_scaled, columns=selected_features)
    df_test["Label"] = y_test.reset_index(drop=True)
    df_test.to_csv("data/test/test_data.csv", index=False)

    return clf, X_test_scaled, y_test, y_pred

# Cho phép chạy độc lập
if __name__ == "__main__":
    df_features = pd.read_csv("data/features/features.csv")  # Bạn cần lưu trước file features từ main
    train_svm(df_features)

