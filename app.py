from flask import Flask
from flask import request
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import supervised_learning as sl
import key_feature_extraction as key


def setup_train():
    train_path = "./train"
    image_paths, images = sl.load_images(train_path)
    image_names = [key.extract_image_name(p) for p in image_paths]
    mapping = sl.create_mapping(image_names)
    labels = sl.create_labels(image_names, mapping)
    samples = key.sample_generator(images)
    model = RandomForestClassifier()
    model.fit(samples, labels)
    inv_mapping = {v: k for k, v in mapping.items()}

    return model, inv_mapping


def read_bytes(raw):
    nparr = np.fromstring(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


model, mapping = setup_train()
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']
    image = read_bytes(file.read())
    sample = key.sample_generator([image])
    result = model.predict(sample)
    label = mapping[result[0]]

    # key.show_img(image)
    # key.pause()

    return label


if __name__ == '__main__':
    app.run(debug=True)
