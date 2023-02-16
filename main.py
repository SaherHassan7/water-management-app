import base64
from sqlite3 import apilevel
from sys import api_version
import numpy as np
import cv2
from flask import Flask, request, jsonify
import tensorflow as tf
from flask import Flask, request, jsonify
import PIL.Image
import os
from flask_restful import Api
import werkzeug
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('watermeter.h5')

app = Flask(__name__)

api = Api(app)

@app.route('/watermeter', methods=['POST'])
def watermeter_image():
    if  request.method=="POST":
        imagefile=request.files['image']
        filename=werkzeug.utils.secure_filename(imagefile.filename)
        img_path="./upload/" + filename
        imagefile.save(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (28, 28))
        image = image/255
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image,axis=0)
        predictions = model.predict(image)
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"prediction":int(np.argmax(predictions))})

if __name__ == '__main__':
      app.run()


