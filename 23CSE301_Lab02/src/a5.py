import pandas as pd
import numpy as np


def load_data():
    file_path = "datasets/Lab Session Data.xlsx"
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def main():

    df = load_data()

    # Select only binary columns (0/1)
    binary_columns = []

    for column in df.columns:
        values = df[column].dropna().unique()

        if set(values).issubset({0, 1}):
            binary_columns.append(column)

    binary_data = df[binary_columns]

    vector1 = binary_data.iloc[0].astype(int).values
    vector2 = binary_data.iloc[1].astype(int).values

    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))

    jaccard = f11 / (f11 + f10 + f01)

    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    print("Binary Columns Used:")
    print(binary_columns)

    print("\nf11 =", f11)
    print("f10 =", f10)
    print("f01 =", f01)
    print("f00 =", f00)

    print("\nJaccard Coefficient :", jaccard)

    print("Simple Matching Coefficient :", smc)


if __name__ == "__main__":
    main()