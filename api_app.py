from flask import Flask, json, request, jsonify, render_template
from detect_api import detect
from werkzeug.utils import secure_filename
import os
import cv2


ai_app = Flask(__name__)

# upload_folder = 'static/uploads'
upload_folder = os.path.join('static', 'uploads')

ai_app.config['UPLOAD'] = upload_folder

@ai_app.route("/", methods=['POST', 'GET'])
def homepage():
    if request.method == 'POST':
        path_image = request.form.get('Iname')
        image_detect = detect(path_image)
        save_name = f"{path_image.split('/')[-1].split('.')[0]}"
        cv2.imwrite(f"static/uploads/{save_name}.jpg", image_detect)
        img = os.path.join(ai_app.config['UPLOAD'], f"{save_name}.jpg")
        # cv2.imshow('Image', img)
        return render_template("homepage.html", pagetitle="Homepage", InameDetect=img)
    
    return render_template("homepage.html", pagetitle="Homepage")

if __name__ == '__main__':
    ai_app.run(debug=True, host='localhost', port=9000)