# ImageProcessing/controller/image_controller.py
from flask import Blueprint, request, jsonify
from service.image_service import process_image

image_bp = Blueprint("image_bp", __name__)

@image_bp.route("/process", methods=["POST"])
def process():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    result_b64 = process_image(data["image"])
    return jsonify({"result": result_b64})