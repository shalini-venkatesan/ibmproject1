# ibmproject1
# Project Title: AI-Based Safety App for Deaf Individuals Using Real-Time Traffic Sound Detection
## Project Description
This project focuses on enhancing the safety of deaf individuals in public spaces by detecting nearby vehicle horns and other traffic sounds using machine learning techniques. The app uses real-time audio processing to classify urban traffic sounds and provides immediate vibratory feedback to users when a potential hazard, such as a vehicle horn, is detected.

## Key Features
1. Detects and classifies vehicle horns and other urban traffic sounds in real-time.
2. Provides vibratory feedback to alert users of nearby potential hazards.
3. Utilizes MFCC (Mel-Frequency Cepstral Coefficients) for feature extraction.
4. Achieves model accuracy through Random Forest and Convolutional Neural Network (CNN) classifiers.
5. User-friendly and responsive frontend developed using HTML, CSS, and JavaScript.
## Technologies Used
1. Machine Learning Framework: Scikit-learn, Keras
2. Audio Processing: Librosa for feature extraction
3. Model: Random Forest and Convolutional Neural Network (CNN)
4. Frontend: HTML, CSS, JavaScript
5. Backend: Python (for model training and predictions)
6. Dataset: ESC-50 (for training) and UrbanSound8K dataset
7. Environment: Google Colab for model training and validation

## program:

```py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0) 

# Define the path to your dataset directory
dataset_dir = '/content/audio_files' 

# Load the relevant classes from a CSV file
relevant_classes = pd.read_csv('/content/esc50.csv') 

# Prepare to store features and labels
X = []  # Feature vectors (MFCCs)
y = []  # Labels

# Check available audio files
available_files = os.listdir(dataset_dir)
print("Available audio files:")
print(available_files)

# Loop through relevant classes to extract MFCCs
# Loop through relevant classes to extract MFCCs
for index, row in relevant_classes.iterrows():
    audio_path = os.path.join(dataset_dir, row['filename'])

    if os.path.exists(audio_path):
        try:
            mfccs = extract_mfcc(audio_path)
            if mfccs is not None and mfccs.shape[0] > 0: 
                X.append(mfccs)
                y.append(row['category'])
            else:
                print(f"MFCC extraction returned no data for: {audio_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Print status every 10 files
    if index % 1000 == 0:
        print(f"Processed {index}/{len(relevant_classes)} files.")


# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# After the loop, check sizes of X and y
print(f"Number of samples in X: {X.shape[0]}")
print(f"Number of samples in y: {y.shape[0]}")

# Proceed with train-test split and model training regardless of data availability
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

```

## output:
![image](https://github.com/user-attachments/assets/3713fb01-64af-43da-a126-828786670cf5)

## front end:
![image](https://github.com/user-attachments/assets/a51e2bfa-14bd-4a9e-97fd-36d8d5213887)
### recording the real-time audio:
![image](https://github.com/user-attachments/assets/eff3a501-3c7a-48ae-bd8b-32ce8ad231d6)

![image](https://github.com/user-attachments/assets/204139c7-e227-4951-9cfd-f8f2e281c943)


## Future Work
Integrate real-time mobile app functionality for on-device sound detection.
Improve the accuracy of the model with larger datasets and more advanced architectures like LSTMs or transformers.
Expand the dataset to cover more diverse environments and sounds.
## Contributors
1. harini
2. beulah
3. rathish
4. bharathraj
5. iyyanar
