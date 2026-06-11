from PIL import Image
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train_processed.csv")
image_path = "dataset/" + df.iloc[0]["full_path"]

img = Image.open(image_path)
print("OG size:", img.size)

img = img.resize((224, 224))
print("Resized Size:", img.size)
img_array = np.array(img)
print("Array Shape:", img_array.shape)