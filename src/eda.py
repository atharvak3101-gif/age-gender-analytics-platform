import pandas as pd

df = pd.read_csv("dataset/train.csv")

bins = [0, 12, 19, 29, 39, 49, 59, 100]

labels = [
    "Child",
    "Teen",
    "Young Adult",
    "Adult",
    "Middle Age",
    "Senior",
    "Elderly"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=bins,
    labels=labels
)

print(df["age_group"].value_counts())