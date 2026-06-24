import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    labels_map = model_dict.get('labels_map', None)
except FileNotFoundError:
    print("Error: Could not find 'model.p'. Please run 'train_model.py' to train and save the model first.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Support 1 hand matching the training data collection
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

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("=== Real-time Sign Language Translator ===")
print("Press 'Q' on the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # We process the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Convert landmarks to pixel space
        landmark_list = calc_landmark_list(w, h, hand_landmarks)
        
        # Apply normalization
        pre_processed_landmarks = pre_process_landmark(landmark_list)

        # Calculate bounding box coordinates
        x_coords = [pt[0] for pt in landmark_list]
        y_coords = [pt[1] for pt in landmark_list]
        x1, y1 = min(x_coords) - 20, min(y_coords) - 20
        x2, y2 = max(x_coords) + 20, max(y_coords) + 20

        # Predict gesture
        if len(pre_processed_landmarks) == 42:
            prediction = model.predict([np.asarray(pre_processed_landmarks)])
            predicted_val = prediction[0]
            
            # Map prediction using labels map if available
            if labels_map is not None:
                try:
                    idx = int(predicted_val)
                    predicted_character = labels_map[idx]
                except (ValueError, TypeError, IndexError):
                    predicted_character = str(predicted_val)
            else:
                predicted_character = str(predicted_val)

            # Draw prediction text and bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                predicted_character, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.3, 
                (0, 255, 0), 
                3, 
                cv2.LINE_AA
            )

    # Show live translation
    cv2.imshow('Sign Language Translator', frame)

    # Exit if 'Q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

