from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/ASUS/Downloads/model (1).tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'})

        # Read image and preprocess
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(image)

        # Set input tensor
        input_data = np.array([preprocessed_image], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke model
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction result
        prediction = np.argmax(output_data, axis=1)

        return jsonify({'prediction': prediction.item()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
