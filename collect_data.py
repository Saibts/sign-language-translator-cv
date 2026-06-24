import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Directory to save the collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Configure the classes and number of samples to collect
classes = ['A', 'B', 'C']  # You can expand this list with more gestures
dataset_size = 100         # Number of samples to collect per class

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Change to 2 if you want to support two-handed signs
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper function to convert landmarks to pixel coordinates
def calc_landmark_list(image_width, image_height, landmarks):
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

# Helper function to normalize landmarks (wrist-relative and scale-normalized)
def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    
    temp_landmark_list = []
    for x, y in landmark_list:
        temp_landmark_list.append(x - base_x)
        temp_landmark_list.append(y - base_y)
        
    # Scale normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value != 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        
    return temp_landmark_list

cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

data = []
labels = []

print("=== Sign Language Data Collection ===")
print(f"We will collect data for classes: {classes}")
print(f"Number of samples per class: {dataset_size}")
print("Press 'Q' to quit at any time.")
print("Press 'S' when you are ready to start capturing each class.\n")

for j, class_name in enumerate(classes):
    print(f"Preparing to collect data for class: '{class_name}'")
    
    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for easier interaction
        frame = cv2.flip(frame, 1)
        
        # Display instructions on screen
        cv2.putText(frame, f"Ready for class '{class_name}'? Press 'S' to Start.", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection - Setup', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            print("Quitting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    cv2.destroyWindow('Data Collection - Setup')

    # Collect samples
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # We process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks on screen for visual feedback
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Convert to pixel space and normalize
            landmark_list = calc_landmark_list(w, h, hand_landmarks)
            pre_processed_landmarks = pre_process_landmark(landmark_list)
                
            # Append normalized keypoints and label
            if len(pre_processed_landmarks) == 42:
                data.append(pre_processed_landmarks)
                labels.append(class_name)
                counter += 1
                
        # Display capture progress
        cv2.putText(frame, f"Collecting: {counter}/{dataset_size}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection - Capturing', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    cv2.destroyWindow('Data Collection - Capturing')
    print(f"Finished collecting data for class: '{class_name}'\n")

cap.release()
cv2.destroyAllWindows()

# Save collected data to a file
dataset_path = os.path.join(DATA_DIR, 'data.pickle')
with open(dataset_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Success! Collected dataset saved to '{dataset_path}'")
print(f"Total samples recorded: {len(data)}")

