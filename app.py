import numpy as np
import os
import random  # You need to import the 'random' module
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)  # Correct the '__name__'

# Load the model
model = load_model("my_model1.h5", compile=False)

# Define the class labels at the global level
class_labels = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Prediction')
def prediction():
    return render_template('Prediction.html')

@app.route('/output', methods=["GET", "POST"])
def output():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)  # Correct '__file__'
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load the image for prediction
        img = image.load_img(filepath, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make a prediction
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        return render_template('output.html', image_name=f.filename, prediction=predicted_label)

    # When the request method is GET or not POST, select a random microbe name
    random_microbe = random.choice(class_labels)
    
    return render_template('output.html', image_name=None, prediction=random_microbe)

if __name__ == '__main__':  # Correct '__name__'
    app.run(debug=True)
