import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset/train")


print(os.listdir(DATA_DIR))
img_path = os.path.join(DATA_DIR, "images")
mask_path = os.path.join(DATA_DIR, "masks")
print(img_path,mask_path)
print("Exists :",os.path.exists(img_path))
