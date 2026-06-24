# Sign Language Translator

A Computer Vision project to translate sign language gestures into text/speech in real-time using OpenCV, MediaPipe, and Scikit-Learn.

## Project Structure
- `download_asl_dataset.py`: Fetch the pre-extracted standard ASL hand landmark dataset (A–Z) from a public repository.
- `collect_data.py`: Capture custom gesture keypoints (hand landmarks) via webcam and save them as datasets.
- `train_model.py`: Train a Random Forest Classifier on either the standard ASL dataset or custom keypoint data.
- `inference.py`: Run real-time translation using webcam and the trained model.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train on Standard ASL Dataset (A–Z) (Recommended):**
   * Download the pre-collected coordinate dataset:
     ```bash
     python download_asl_dataset.py
     ```
   * Train the classifier:
     ```bash
     python train_model.py
     ```
     This trains on the 36,400+ hand landmarks and achieves **~94% validation accuracy** across all 26 alphabets.

3. **Or, Collect Custom Data:**
   * Run the custom data collection script:
     ```bash
     python collect_data.py
     ```
     Follow the prompts to record custom landmarks via webcam.
   * Train the model on your custom dataset:
     ```bash
     python train_model.py
     ```

4. **Run Real-Time Inference:**
   Start the real-time sign language translator:
   ```bash
   python inference.py
   ```
   Show hand gestures in front of the webcam to translate them to letters on screen. Press **`Q`** to quit.

