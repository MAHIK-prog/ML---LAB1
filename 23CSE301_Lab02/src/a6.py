import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    file_path = "datasets/Lab Session Data.xlsx"
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def main():

    df = load_data()

    encoder = LabelEncoder()

    # Handle every column
    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())

        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            df[column] = encoder.fit_transform(df[column].astype(str))

    vector1 = df.iloc[0].values.reshape(1, -1)
    vector2 = df.iloc[1].values.reshape(1, -1)

    cosine = cosine_similarity(vector1, vector2)[0][0]

    print("\nCosine Similarity between Observation 1 and Observation 2")
    print(cosine)


if __name__ == "__main__":
    main()