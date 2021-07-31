from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))


# DONT FORGET TO USE ESCAPE()

def chooseAnimation(gesture):
    animations = {
        "01_palm": "palm",
        "02_l": "L",
        "03_fist": "fist",
        "04_fist_moved": "fistMove",
        "05_thumb": "thumb",
        "06_index": "index",
        "07_ok": "ok",
        "08_palm_moved": "palmMove",
        "09_c": "C",
        "10_down": "downwards"
    }

    gestures = {
        "01_palm": "Open palm",
        "02_l": "Fingers in the shape of an L",
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

    [animation, gesture] = chooseAnimation("01_palm")

    html = render_template("index.html", animation=animation, gesture=gesture, certainty="99.9%")
    return html


@app.route('/predict', methods=['POST'])
def predict():
    animation = chooseAnimation("09_c")

    return render_template('index.html', gesture=animation, animation=animation, certainty="85.0%")
