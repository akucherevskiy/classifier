import base64
import io

import cv2
import flask
import numpy as np
from PIL import Image
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array

app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = ResNet50(weights="imagenet")


def remove_white_background(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)

    return image[y:y + h, x:x + w]


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def get_white_pixels_percentage(image):
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    colors, counts = np.unique(im_arr.reshape(-1, 3), axis=0, return_counts=True)
    white_pixels_count = all_count = 0
    for color, count in zip(colors, counts):
        if (color > 250).all():
            white_pixels_count += count
        all_count += count
    return white_pixels_count / all_count * 100


@app.route("/classify", methods=["POST"])
def predict():
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.form.get('image'):
            image = Image.open(io.BytesIO(base64.b64decode(flask.request.form.get('image'))))

            image = remove_white_background(image)
            white_pixels_percentage = get_white_pixels_percentage(image)
            image = prepare_image(Image.fromarray(image, 'RGB'), target=(224, 224))

            plan_words = ['menu', 'crossword_puzzle', 'web_site', 'envelope', 'comic_book']
            predictions = imagenet_utils.decode_predictions(model.predict(image))
            is_plan = 0
            if white_pixels_percentage > 15:
                is_plan += 1
            data["predictions"] = []
            for (imagenetID, label, prob) in predictions[0]:
                if label in plan_words and prob > 0.1:
                    is_plan += 1
            data['data'] = []
            data['data'].append({'is_plan': is_plan > 0})
            data['success'] = True
            return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(threaded=True)

# uwsgi --socket 0.0.0.0:5000 --protocol=http -w classifier:app -p 4
