import pandas as pd
def define_age_groups(df):
    bins = [0, 19, 29, 39, 49, 59, 100]

    labels = [
        "Young",
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior",
        "Elderly"
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df