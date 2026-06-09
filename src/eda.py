import pandas as pd
from utils import define_age_groups
df = pd.read_csv("dataset/train.csv")

df = define_age_groups(df)