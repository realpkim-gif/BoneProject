import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras

IMG_SIZE = 224

model = keras.models.load_model("bone_fracture_model.keras")

# Load one image
img_path = "BoneData/BoneFractureYolo8/valid/images"
fn = os.listdir(img_path)[0]
img = Image.open(os.path.join(img_path, fn)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
img = np.asarray(img) / 255.0

# Load its label
lbl_path = "BoneData/BoneFractureYolo8/valid/labels"
with open(os.path.join(lbl_path, fn.replace(".jpg", ".txt"))) as f:
    label = [float(v) for v in f.readline().split()[1:9]]

# Predict (model needs a batch, so wrap the single image in a batch of 1)
img_batch = np.expand_dims(img, axis=0)   # shape (224, 224, 3) -> (1, 224, 224, 3)
prediction = model.predict(img_batch)     # shape (1, 8)
pred = prediction[0]                      # shape (8,) -- pull the one result out of the batch


# Plot
fig, axes = plt.subplots(1, 2)

axes[0].imshow(img)
axes[0].set_title("Label")
for i in range(0, 8, 2):
    axes[0].plot(label[i] * IMG_SIZE, label[i+1] * IMG_SIZE, "go")

axes[1].imshow(img)
axes[1].set_title("Prediction")
for i in range(0, 8, 2):
    axes[1].plot(pred[i] * IMG_SIZE, pred[i+1] * IMG_SIZE, "ro")

plt.show()
