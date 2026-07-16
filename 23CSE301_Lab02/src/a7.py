import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    file_path = "datasets/Lab Session Data.xlsx"
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def preprocess(df):

    encoder = LabelEncoder()

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            df[column] = encoder.fit_transform(df[column].astype(str))

    return df


def jaccard(v1, v2):

    v1 = (v1 > 0).astype(int)
    v2 = (v2 > 0).astype(int)

    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))

    denominator = f11 + f10 + f01

    if denominator == 0:
        return 0

    return f11 / denominator


def smc(v1, v2):

    v1 = (v1 > 0).astype(int)
    v2 = (v2 > 0).astype(int)

    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))

    return (f11 + f00) / (f11 + f10 + f01 + f00)


def main():

    df = load_data()
    df = preprocess(df)

    df = df.iloc[:20]

    n = len(df)

    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            v1 = df.iloc[i].values
            v2 = df.iloc[j].values

            jc_matrix[i][j] = jaccard(v1, v2)
            smc_matrix[i][j] = smc(v1, v2)
            cosine_matrix[i][j] = cosine_similarity(
                [v1], [v2]
            )[0][0]

    plt.figure(figsize=(8, 6))
    sns.heatmap(jc_matrix, annot=True, cmap="Blues")
    plt.title("Jaccard Coefficient Heatmap")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(smc_matrix, annot=True, cmap="Greens")
    plt.title("Simple Matching Coefficient Heatmap")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_matrix, annot=True, cmap="Reds")
    plt.title("Cosine Similarity Heatmap")
    plt.show()


if __name__ == "__main__":
    main()