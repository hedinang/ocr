from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
from pdf2image import convert_from_path
from datetime import datetime
from process import Process
app = Flask(__name__)

process = Process('cuda')


@app.route('/hello', methods=['GET'])
def get():
    return 'Hello this is Legal Document Ocr Application'


@app.route('/process', methods=['POST'])
def processUrl():
    result = []
    if request.method == 'POST':
        f = request.files['img']
        filename = f.filename
        img_name = filename
        f.save(secure_filename(filename))
        if 'pdf' in filename.lower():
            filename.split('.')[0]
            img_name = '{}.png'.format(filename.split('.')[0])
            pages = convert_from_path(
                filename, 500, size=1200)
            pages[0].save(img_name, 'png')
        result = process(img_name)
        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
