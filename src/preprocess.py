import pandas as pd
from utils import define_age_groups
from sklearn.preprocessing import LabelEncoder #type: ignore
df = pd.read_csv("dataset/train.csv")

df = define_age_groups(df)
age_encoder = LabelEncoder()

df["age_group_encoded"] = age_encoder.fit_transform(df["age_group"])

print(df[["age", "age_group", "age_group_encoded"]].head())
df.to_csv("dataset/processed_train.csv", index=False)