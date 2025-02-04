import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from pycocotools.coco import COCO
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#load in coco
COCO_PATH = "coco2017/annotations/instances_train2017.json"
IMAGES_PATH = "coco2017/train2017/"

print("Loading COCO dataset...")
coco = COCO(COCO_PATH)
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_names = [cat["name"] for cat in categories]
label_encoder = LabelEncoder()
label_encoder.fit(category_names)

#hog features
def extract_hog_features(image, bbox):
    x, y, w, h = map(int, bbox)
    cropped = image[y:y+h, x:x+w]
    
    if cropped.size == 0:
        return None
    
    cropped = cv2.resize(cropped, (64, 128))#resize to standard
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features

X, y = [], []

#adding progress bar to see what i'm doing
image_ids = coco.getImgIds()[:1000]#only using 1000 images because dataset too large otherwise

print(f"Processing {len(image_ids)} images...")
for img_id in tqdm(image_ids, desc="Extracting Features"):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGES_PATH, img_info["file_name"])
    
    image = cv2.imread(img_path)
    if image is None:
        continue

    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)
    
    for ann in annotations:
        bbox = ann["bbox"]
        category_id = ann["category_id"]
        category_name = coco.loadCats([category_id])[0]["name"]

        features = extract_hog_features(image, bbox)
        if features is not None:
            X.append(features)
            y.append(category_name)

#svm
if not X:
    raise ValueError("No features extracted! Check dataset paths.")

X = np.array(X)
y = label_encoder.transform(y)

print("Splitting dataset for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SGDClassifier(loss="hinge", max_iter=1, warm_start=True)

num_epochs = 20
print("Training SVM classifier...")

for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    svm.partial_fit(X_train, y_train, classes=np.unique(y))

print("SVM training complete!")

joblib.dump((svm, label_encoder), "hog_svm_coco.pkl")
print("Model saved as 'hog_svm_coco.pkl'")