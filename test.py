import os
import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df, base_path):
    X = []
    y = []
    for index, row in df.iterrows():
        image_path = '/Users/bobby4/Desktop/final/anger'
        image_path = os.path.join(base_path, row['pth'])
        print(f"Loading image from: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read image at path: {image_path}")
            continue
        
        # Resize the image to a standard size (e.g., 224x224)
        image = cv2.resize(image, (224, 224))
        
        # Flatten the image (convert 2D matrix to 1D array)
        image = image.flatten()
        
        X.append(image)
        y.append(row['label'])
    
    return X, y

# Load CSV data
df = pd.read_csv('/Users/bobby4/Desktop/final/labels.csv')

# Base path where the images are located
base_path = '/Users/bobby4/Desktop/final/'

# Preprocess the data
X, y = preprocess_data(df, base_path)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# To use the model for predictions:
def predict_emotion(image_path):
    # Load the trained model and label encoder
    model = joblib.load('emotion_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image
    image = image.flatten()  # Flatten the image
    
    # Predict the emotion
    prediction = model.predict([image])
    predicted_label = label_encoder.inverse_transform(prediction)
    
    return predicted_label[0]

# Example usage:
image_path = '/Users/bobby4/Desktop/final/anger/image0000060.jpg'  # Path to a new image
predicted_expression = predict_emotion(image_path)
print(f"Predicted expression: {predicted_expression}")
