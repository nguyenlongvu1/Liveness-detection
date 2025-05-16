import joblib
import pandas as pd
from sklearn.metrics import classification_report
from visualize import plot_confusion_matrix

def evaluate_model(model_path="models/svm_model.pkl", scaler_path="models/scaler.pkl", test_data_path="data/test/test_data.csv"):
    # Load dữ liệu test
    df = pd.read_csv(test_data_path)
    X_test = df[["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"]]
    y_test = df["Label"]

    # Load model và scaler
    clf = joblib.load(model_path)

    # Scale dữ liệu test
    X_scaled = X_test

    # Dự đoán
    y_pred = clf.predict(X_scaled)

    # Báo cáo
    print(classification_report(y_test, y_pred))

    # Trực quan
    plot_confusion_matrix(y_test, y_pred)
    #plot_tsne(X_scaled, y_pred)

if __name__ == "__main__":
    evaluate_model()

