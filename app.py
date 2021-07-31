from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)


def createModel():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(180,
                                                                      180,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    modelBuilt = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10)
    ])

    modelBuilt.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return modelBuilt


model = createModel()
model.load_weights("model_weights.h5")


# DONT FORGET TO USE ESCAPE()

def chooseAnimation(gesture):
    animations = {
        "None": "NoAnimation",
        "01_palm": "Palm",
        "02_l": "L",
        "03_fist": "Fist",
        "04_fist_moved": "FistMove",
        "05_thumb": "Thumb",
        "06_index": "Index",
        "07_ok": "Ok",
        "08_palm_moved": "PalmMove",
        "09_c": "C",
        "10_down": "Downwards"
    }

    gestures = {
        "None": "No gesture",
        "01_palm": "Open palm",
        "02_l": "L shape",
        "03_fist": "Closed fist",
        "04_fist_moved": "A fist moving",
        "05_thumb": "Thumb up",
        "06_index": "Index up",
        "07_ok": "ok sign",
        "08_palm_moved": "Open palm moving",
        "09_c": "C shape",
        "10_down": "Get down sign"
    }

    return [animations[gesture], gestures[gesture]]


@app.route("/", methods=['GET', 'POST'])
def baseHtml():

    [animation, gesture] = chooseAnimation("None")

    html = render_template("index.html", animation=animation, gesture=gesture, certainty="N/A")
    return html


@app.route('/predict', methods=['POST'])
def predict():

    [animation, gesture] = chooseAnimation("03_fist")

    return render_template('index.html', animation=animation, gesture=gesture, certainty="85.0%")
