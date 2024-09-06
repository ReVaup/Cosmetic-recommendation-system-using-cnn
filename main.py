import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Define the path for the uploads folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

# Load the model
model = load_model('./minor_project/Skin_type_determination.h5')

# Load the cosmetics dataset
try:
    cosmetics_df = pd.read_csv('C:/Users/reVaup/Desktop/programs/minor project/cosmetics csv/cosmetics.csv')
except PermissionError:
    print("Permission denied: 'C:/Users/reVaup/Desktop/programs/minor project/cosmetics csv/cosmetics.csv'")
    cosmetics_df = None
except FileNotFoundError:
    print("File not found: 'C:/Users/reVaup/Desktop/programs/minor project/cosmetics csv/cosmetics.csv'")
    cosmetics_df = None
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    cosmetics_df = None

def predict_skin_type(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)
    class_labels = ['Combination', 'Dry', 'Oily']
    return class_labels[predicted_class_idx[0]]

def recommend_products_for_skin_type(skin_type, df):
    skin_type = skin_type.capitalize()
    recommended = df[df[skin_type] == 1]
    top_recommendations = recommended.sort_values(by='Rank', ascending=False)
    return top_recommendations

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            skin_type = predict_skin_type(img_path)
            recommendations = recommend_products_for_skin_type(skin_type, cosmetics_df)
            recommendations_html = recommendations.to_html(classes='table') if isinstance(recommendations,pd.DataFrame) else ""
            return render_template('result.html', skin_type=skin_type, recommendations=recommendations_html)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

