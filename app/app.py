# coding:utf-8
import os
import sys

BASE_DIR = os.path.dirname(os.getcwd())
sys.path.append(BASE_DIR)

from flask_cors import cross_origin
from PIL import Image
import settings
import helpers
import flask
import redis
import uuid
import json
import io
import time

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            try:
                image = Image.open(io.BytesIO(image))
            except IOError:
                data["msg"] = "服务器无法解析图片"
            else:
                image = prepare_image(image)

                k = str(uuid.uuid4())

                # base64 encode str
                image = helpers.base64_encode_image(image)

                d = {"id": k, "image": image}
                db.rpush(settings.DET_QUEUE, json.dumps(d))
                while True:
                    output = db.get(k)
                    if output is not None:
                        data["result"] = json.loads(output)
                        db.delete(k)
                        break
                    time.sleep(settings.CLIENT_SLEEP)
                data["success"] = True
    return flask.jsonify(data)

@app.route("/predict-rela", methods=["POST"])
@cross_origin()
def predict_rela():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            try:
                image = Image.open(io.BytesIO(image))
            except IOError:
                data["msg"] = "服务器无法解析图片"
            else:
                image = prepare_image(image)

                k = str(uuid.uuid4())

                # base64 encode str
                image = helpers.base64_encode_image(image)

                d = {"id": k, "image": image}
                db.rpush(settings.RELA_QUEUE, json.dumps(d))
                while True:
                    output = db.get(k)
                    if output is not None:
                        data["result"] = json.loads(output)
                        db.delete(k)
                        break
                    time.sleep(settings.CLIENT_SLEEP)
                data["success"] = True
    return flask.jsonify(data)

@app.route("/", methods=["GET"])
@cross_origin()
def hello():
    return "Hello, world"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True)
