from flask import Flask, url_for, render_template

app = Flask(__name__)

#styleCss = url_for('static', filename='style.css')


# DONT FORGET TO USE ESCAPE()


@app.route("/", methods=['GET', 'POST'])
def baseHtml():
    animations = {
        "left": "aliveLeft",
        "right": "aliveRight"
    }

    animation = animations["left"]

    html = render_template("index.html", animation=animation, gesture="Lefted", certainty="99.9%")
    return html
