import numpy as np
import pandas as pd
import os
import ultralytics
from ultralytics import YOLO

model = YOLO("yolo26n")
results = model.train(data="BoneData/BoneFractureYolo8/data.yaml", epochs=20, imgsz=640)
