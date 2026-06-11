from PIL import Image
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train_processed.csv")

def create_pipeline(df):
    X_path = []
    y_age = []
    y_gender = []

    for _,row in df.iterrows():
        # image
        img_path = "dataset/" + row["full_path"]
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        X_path.append(img_array)

        # age & gender
        y_age.append(row["age_group_encoded"])
        y_gender.append(row["gender"])

    return (np.array(X_path), np.array(y_age), np.array(y_gender))


sample_df = df.head(10)

X, y_age, y_gender = create_pipeline(sample_df)

print(X.shape)
print(y_age.shape)
print(y_gender.shape)