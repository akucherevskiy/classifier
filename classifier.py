import base64
import cv2
import flask
import io
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def prepare_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def get_white_pixels_percentage(image):
    colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    white_pixels_count = all_count = 0
    for color, count in zip(colors, counts):
        if (color > 250).all():
            white_pixels_count += count
        all_count += count
    return white_pixels_count / all_count * 100


@app.route("/classify", methods=["POST"])
def predict():
    data = {"success": False, 'data': {'is_plan': False, 'error': ''}}
    if flask.request.form.get('image'):
        image = Image.open(io.BytesIO(base64.b64decode(flask.request.form.get('image'))))

        im_arr = np.array(image)
        image = remove_white_background(im_arr)

        white_pixels_percentage = get_white_pixels_percentage(image)

        predictions = imagenet_utils.decode_predictions(model.predict(prepare_image(Image.fromarray(image))))

        is_plan = False
        if white_pixels_percentage > 15:
            is_plan += True
        plan_words = ['menu', 'crossword_puzzle', 'web_site', 'envelope', 'comic_book']
        for (imagenetID, label, prob) in predictions[0]:
            if label in plan_words and prob > 0.1:
                is_plan += True
        if is_plan:
            data['data']['is_plan'] = is_plan
        data['success'] = True

        return flask.jsonify(data)
    else:
        data['data']['error'] = 'Bad Request'

        return flask.jsonify(data)


if __name__ == "__main__":
    print("* Loading classifier model...")
    load_model()
    app.run(threaded=False)
