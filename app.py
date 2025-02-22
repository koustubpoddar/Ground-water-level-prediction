                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import joblib
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64


print("Current working directory:", os.getcwd())


app = Flask(__name__)

# Load the trained model
model = joblib.load('Trained_model.pkl')

# Create a directory to store the output images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

@app.route('/')
def index():
     return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    csv_file = request.files['csv_file']
    df = pd.read_csv(csv_file)

    # Preprocess the data
    df = df.drop(['elec_pos'], axis=1)  # Remove explanatory variables
    df = df.dropna()  # Remove rows with missing values

    # Predict the groundwater presence
    predictions = model.predict(df)

    # Generate the image
    image_name = '2d_subsurface_profile_presence.png'
    generate_image(df, predictions, image_name)

    # Return the image name as a JSON response
    return jsonify({'image_name': image_name, 'predictions': predictions.tolist()})

def generate_image(df, predictions, image_name):
    # Generate the image using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(df['depth'], df['rho'], label='Resistivity')
    plt.plot(df['depth'], predictions, label='Predicted Groundwater Presence')
    plt.xlabel('Depth (m)')
    plt.ylabel('Resistivity (ohm-m)')
    plt.title('2D Subsurface Profile')
    plt.legend()

    # Return the image data as a base64 encoded string
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img_data_b64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    return img_data_b64

@app.route('/output_images/<string:image_name>')
def serve_image(image_name):
    return send_from_directory(output_dir, image_name, as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)