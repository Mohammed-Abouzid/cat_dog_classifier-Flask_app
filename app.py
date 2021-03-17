from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import numpy as np
import cv2
# ML Packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Loading our ML Model
    classifier = load_model('./model/cat_dog_classifier.h5')
    # Receives the input query from form
    if request.method == 'POST':
        namequery = request.form['namequery']
        test_image = image.load_img(namequery, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        my_prediction= int(result[0][0])
    return render_template('results.html',prediction = my_prediction, name = namequery.upper())
if __name__ == '__main__':
    app.run(debug=True)
