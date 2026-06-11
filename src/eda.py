import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("dataset/processed_train.csv")


train_df, validation_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["age_group"]
)

train_df.to_csv("dataset/train_proceessed.csv", index=False)
validation_df.to_csv("dataset/validation_processed.csv", index=False)