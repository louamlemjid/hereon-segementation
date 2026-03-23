# ImageProcessing/main.py
from flask import Flask
from controller.image_controller import image_bp

app = Flask(__name__)
app.register_blueprint(image_bp, url_prefix="/image")
@app.route("/")
def home():
    return "Hello from flask app"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)