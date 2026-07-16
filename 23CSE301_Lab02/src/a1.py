import pandas as pd
import numpy as np


def load_data():
    file_path = "datasets/Lab Session Data.xlsx"
    return pd.read_excel(file_path, sheet_name="Purchase data")


def create_matrices(df):
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
    y = df["Payment (Rs)"].to_numpy()
    return X, y


def main():
    df = load_data()

    X, y = create_matrices(df)

    print("Purchase Data")
    print(df)

    print("\nFeature Matrix (X):")
    print(X)

    print("\nOutput Vector (y):")
    print(y)

    print("\nDimensionality of Vector Space :", X.shape[1])

    print("Number of Vectors :", X.shape[0])

    rank = np.linalg.matrix_rank(X)
    print("Rank of Feature Matrix :", rank)

    pseudo_inverse = np.linalg.pinv(X)

    product_cost = pseudo_inverse @ y

    print("\nEstimated Cost of Products")
    print(f"Candies (#)      : Rs. {product_cost[0]:.2f}")
    print(f"Mangoes (Kg)     : Rs. {product_cost[1]:.2f}")
    print(f"Milk Packets (#) : Rs. {product_cost[2]:.2f}")


if __name__ == "__main__":
    main()