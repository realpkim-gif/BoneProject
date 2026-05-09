import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

for item in os.listdir("BoneData/BoneFractureYolo8/train"):
    print(item)

train_label_folder = "BoneData/BoneFractureYolo8/train/labels"
train_rows = []

for filename in os.listdir(train_label_folder):
    filepath = os.path.join(train_label_folder, filename)
    with open(filepath, 'r') as f:
        for line in f:
            values = line.strip().split()
            train_rows.append([filename] + values)

train_df = pd.DataFrame(train_rows)
train_df = train_df.sort_values(by=1)  # sort by first value (class id)
print(train_df.shape)

val_rows=[]
val_label_folder = "BoneData/BoneFractureYolo8/valid/labels"
for filename in os.listdir(val_label_folder):
    filepath = os.path.join(val_label_folder, filename)
    with open(filepath, 'r') as f:
        for line in f:
            values = line.strip().split()
            val_rows.append([filename] + values)
v_df = pd.DataFrame(val_rows)
v_df= v_df.sort_values(by=1)



from PIL import Image

IMG_SIZE = 224
train_img_folder = "BoneData/BoneFractureYolo8/train/images"
val_img_folder   = "BoneData/BoneFractureYolo8/valid/images"

# build a quick lookup: filename stem -> full image filename (handles .jpg/.png/etc)
def index_images(folder):
    out = {}
    for fn in os.listdir(folder):
        stem = os.path.splitext(fn)[0]
        out[stem] = fn
    return out

train_img_index = index_images(train_img_folder)
val_img_index   = index_images(val_img_folder)

# turn a label dataframe + image folder into (X, Y) arrays
# df columns: 0 = "<stem>.txt", 1 = class, 2..9 = 8 OBB coords (as strings)
def df_to_arrays(df, img_folder, img_index, size=IMG_SIZE):
    X, Y = [], []
    seen = set()
    for row in df.itertuples(index=False):
        stem = os.path.splitext(row[0])[0]
        if stem in seen:           # one image per row -> use only first label
            continue
        if stem not in img_index:  # skip if image is missing
            continue
        seen.add(stem)

        coords = [float(v) for v in row[2:10]]  # 8 numbers
        img_path = os.path.join(img_folder, img_index[stem])
        img = Image.open(img_path).convert("RGB").resize((size, size))

        X.append(np.asarray(img, dtype=np.float32) / 255.0)
        Y.append(coords)
    return np.stack(X), np.array(Y, dtype=np.float32)

X_train, y_train = df_to_arrays(train_df, train_img_folder, train_img_index)
X_val,   y_val   = df_to_arrays(v_df,     val_img_folder,   val_img_index)

print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   " y_val:  ", y_val.shape)

"""
END OF PREPROCESSING
"""

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(8, activation="sigmoid"),  # 8 OBB coords, all in [0,1]
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()



from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',    # watch validation loss
    patience=5,            # stop after 5 epochs of no improvement
    restore_best_weights=True  # revert to the best model when it stops
)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=32,callbacks=[early_stop]
)


model.save("bone_fracture_model.keras")

