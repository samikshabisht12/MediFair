import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest


def label_price(score):
    if score > 1.2:
        return "Overpriced"
    elif score < 0.8:
        return "Underpriced"
    else:
        return "Fair"


def main():
    # Load data
    df = pd.read_csv("data/hospital_prices.csv")
    print("Initial Data:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nData Description:")
    print(df.describe())

    # Basic analysis
    avg_price = df.groupby("service")["price"].mean()
    print("\nAverage price per service:")
    print(avg_price)

    print("\nMissing values check:")
    print(df.isnull().sum())

    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=["city", "service"])
    print("\nEncoded data preview:")
    print(df_encoded.head())

    df_encoded = df_encoded.drop("hospital", axis=1)
    
    # Model training preparation
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open("price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nModel saved to price_model.pkl")

    # Evaluate model
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    print("\nMean Absolute Error on test set:", error)

    # Calculate fairness on entire dataset
    df_encoded["predicted_price"] = model.predict(X)
    df_encoded["fairness_score"] = df_encoded["price"] / df_encoded["predicted_price"]
    df_encoded["price_label"] = df_encoded["fairness_score"].apply(label_price)

    print("\nFinal Fairness results preview:")
    print(df_encoded[["price", "predicted_price", "fairness_score", "price_label"]].head())

    # Anomaly Detection using Isolation Forest
    iso_model = IsolationForest(contamination=0.2, random_state=42)
    iso_model.fit(df_encoded[["price", "predicted_price"]])

    df_encoded["anomaly_flag"] = iso_model.predict(df_encoded[["price", "predicted_price"]])
    df_encoded["anomaly_label"] = df_encoded["anomaly_flag"].apply(
        lambda x: "Anomalous" if x == -1 else "Normal"
    )

    print("\nAnomaly Detection Results preview:")
    print(df_encoded[["price", "predicted_price", "price_label", "anomaly_label"]].head())

    # Anomalies
    anomalies = df_encoded[df_encoded["anomaly_label"] == "Anomalous"]
    print("\nSuspicious Pricing records:")
    print(anomalies[["price", "predicted_price", "price_label"]])

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(df_encoded["price"], df_encoded["predicted_price"])
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Price vs Predicted Price")
    plt.show()

    price_label_counts = df_encoded["price_label"].value_counts()
    plt.figure(figsize=(6, 4))
    price_label_counts.plot(kind="bar")
    plt.xlabel("Price Category")
    plt.ylabel("Count")
    plt.title("Distribution of Price Fairness")
    plt.show()


if __name__ == "__main__":
    main()
