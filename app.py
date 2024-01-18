import flask
from flask import request, render_template
import numpy as np
from keras.models import load_model
from PIL import Image
from flask import redirect
import json
import os
from werkzeug.utils import secure_filename

app = flask.Flask(__name__, template_folder='.')
model = load_model("animal_mn.h5")

with open('clases.json', 'r') as json_file:
    classes_data = json.load(json_file)

classes = classes_data.get('clases', [])
model.classes_ = classes

initial_image_path = "static/image_placeholder.jpg"

def predict_animal(image):
    # Cargar la imagen como un array de NumPy
    img = Image.open(image)
    # Redimensionar la imagen a (224, 224)
    img = img.resize((224, 224))
    img_array = np.array(img)
    # Convertir la imagen a un tensor
    img_tensor = np.expand_dims(img_array, axis=0)
    # Realizar la predicción
    prediction = model.predict(img_tensor)    
    # Obtener el índice de la clase con la probabilidad más alta
    class_id = np.argmax(prediction)
    # Obtener el nombre de la clase
    class_name = model.classes_[class_id]
    # class_name = class_id
    return class_name

@app.route("/", methods=["GET", "POST"])
def index():
    class_name = None
    selected_image_path = initial_image_path

    if request.method == "POST":
        image = request.files["image"]
        if image.filename != '':
            filename = "image_posted" + os.path.splitext(image.filename)[1]
            selected_image_path = os.path.join("static", filename)
            image.save(selected_image_path)
            class_name = predict_animal(selected_image_path)

    return render_template("index.html", class_name=class_name, selected_image_path=selected_image_path)


if __name__ == "__main__":
    app.run(debug=True)
