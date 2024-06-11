import requests
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)

API_KEY = 'oMxgGvvksMF9id5rTuKibw==WDetfgRVPSTKiV9k'

# Initialize global variables to store the response
prediction_result = {"class": "", "confidence_score": 0.0, "class_info": {}}

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


def process_and_predict(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict the class
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        class_name = class_name[2:].strip()  # Extract the class name from the model's class_names

        # Get information from the external API based on the predicted class name
        api_url = f'https://api.api-ninjas.com/v1/dogs?name={class_name}'
        headers = {'X-Api-Key': API_KEY}

        response = requests.get(api_url, headers=headers)

        if response.status_code == requests.codes.ok:
            class_info = response.json()
            return {"class": class_name, "confidence_score": float(confidence_score), "class_info": class_info}
        else:
            return {"class": class_name, "confidence_score": float(confidence_score),
                    "class_info": "Error fetching info"}
    except Exception as e:
        return {"class": "Error", "confidence_score": 0.0, "class_info": "Error fetching info"}


@app.route('/process_image', methods=['POST'])
def process_image():
    global prediction_result

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided in the request."})

        image = request.files['image']

        if image.filename == "":
            return jsonify({"error": "No file selected."})

        # Read the uploaded image and predict the class
        prediction_result = process_and_predict(image.read())

        return jsonify({"class": prediction_result["class"], "confidence_score": prediction_result["confidence_score"]})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/get_response', methods=['GET'])
def get_response():
    global prediction_result
    return jsonify({
        "class": prediction_result["class"],
        "confidence_score": prediction_result["confidence_score"],
        "class_info": prediction_result["class_info"]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
