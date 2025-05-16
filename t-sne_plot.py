import pandas as pd
import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tsne_from_features(feature_path="data/features/features.csv",
                            scaler_path="models/scaler.pkl",
                            label_col="Label"):
    # Load features
    df = pd.read_csv(feature_path)

    # Chọn đặc trưng và nhãn
    features = ["AIC", "BIC", "ResidualVar", "ResidualMean", "ResidualStd"]
    X = df[features].dropna()
    y = df[label_col].iloc[X.index]

    # Scale nếu cần (đã dùng trong SVM)
    try:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    except:
        print("[!] Không tìm thấy scaler, sử dụng dữ liệu gốc.")
        X_scaled = X

    # Giảm chiều bằng t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Vẽ đồ thị
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("hsv", len(set(y)))

    for i, label in enumerate(sorted(set(y))):
        idx = (y == label)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Label {label}", alpha=0.7)

    plt.legend()
    plt.title("t-SNE visualization of SARIMA features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_tsne_from_features()
