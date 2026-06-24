import os
import urllib.request

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# URLs to the pre-extracted ASL hand landmark dataset (A-Z)
CSV_URL = "https://raw.githubusercontent.com/AkramOM606/American-Sign-Language-Detection/main/model/keypoint_classifier/keypoint.csv"
LABEL_URL = "https://raw.githubusercontent.com/AkramOM606/American-Sign-Language-Detection/main/model/keypoint_classifier/keypoint_classifier_label.csv"

csv_path = os.path.join(DATA_DIR, 'keypoint.csv')
label_path = os.path.join(DATA_DIR, 'keypoint_classifier_label.csv')

print("Downloading standard ASL landmark dataset...")
try:
    print(f"Downloading CSV from: {CSV_URL}")
    urllib.request.urlretrieve(CSV_URL, csv_path)
    print(f"Saved to {csv_path}")

    print(f"Downloading labels from: {LABEL_URL}")
    urllib.request.urlretrieve(LABEL_URL, label_path)
    print(f"Saved to {label_path}")
    
    print("\nDownload complete! You can now run train_model.py to train the model on this dataset.")
except Exception as e:
    print(f"Error downloading files: {e}")
