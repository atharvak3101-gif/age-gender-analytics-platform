import tensorflow as tf # type: ignore
import numpy as np
import pandas as pd

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def process(image_path, age, gender):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, {
        "age": age,
        "gender": tf.cast(gender, tf.float32)
    }

def create_dataset(df, shuffle=False):
    image_paths = [
        "dataset/" + path
        for path in df["full_path"]
    ]

    ages = df["age_group_encoded"].values
    genders = df["gender"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, ages, genders)
    )

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_df = pd.read_csv("dataset/train_processed.csv")
val_df = pd.read_csv("dataset/validation_processed.csv")

train_dataset = create_dataset(train_df, shuffle=True)
val_dataset = create_dataset(val_df)

for images, labels in train_dataset.take(1):
    print("Images:", images.shape)
    print("Age:", labels["age"].shape)
    print("Gender:", labels["gender"].shape)