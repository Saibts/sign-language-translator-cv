import os
import pickle
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_DIR = './data'
csv_path = os.path.join(DATA_DIR, 'keypoint.csv')
label_path = os.path.join(DATA_DIR, 'keypoint_classifier_label.csv')
pickle_path = os.path.join(DATA_DIR, 'data.pickle')

# Load the dataset
if os.path.exists(csv_path):
    print("Using downloaded standard ASL landmark dataset (keypoint.csv)...")
    
    # Load landmarks (columns 1 to 42) and labels (column 0)
    # The range 1 to 43 corresponds to index 1 through 42 (21 landmarks * 2 coordinates)
    x_data = np.loadtxt(csv_path, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
    y_data = np.loadtxt(csv_path, delimiter=',', dtype='int32', usecols=0)
    
    # Load alphabet labels
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            labels_map = [row[0].strip() for row in reader if row]
    else:
        # Fallback if label file doesn't exist
        labels_map = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
    print(f"Loaded {len(x_data)} samples with {len(labels_map)} classes.")
    
elif os.path.exists(pickle_path):
    print("Using custom dataset (data.pickle)...")
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    x_data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    
    # Convert text labels to unique classes and map them to integers
    unique_labels = sorted(list(set(labels)))
    labels_map = unique_labels
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    y_data = np.array([label_to_id[label] for label in labels], dtype='int32')
    
    print(f"Loaded {len(x_data)} custom samples with classes: {labels_map}")
else:
    print("Error: No dataset found. Please run collect_data.py or download_asl_dataset.py first.")
    exit()

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, shuffle=True, stratify=y_data, random_state=42
)

# Train Classifier
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"\nModel Accuracy: {score * 100:.2f}%")

print("\nClassification Report:")
target_names = [str(label) for label in labels_map]
print(classification_report(y_test, y_predict, target_names=target_names))

# Save the model and labels map together
model_path = './model.p'
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'labels_map': labels_map}, f)

print(f"Success! Model trained and saved to '{model_path}'")

