from flask import Blueprint, request, current_app, render_template, jsonify

from .ml.tester import ImageTester

api = Blueprint("api", __name__, url_prefix="/api")


@api.route("/infer", methods=["POST"])
def post_image():
    img = request.files.get('image')
    img_tester = ImageTester()
    result = img_tester.test(img)
    return jsonify(result)


@api.route("/display")
def render_display():
    return render_template("home/index.html")
