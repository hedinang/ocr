from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from pdf2image import convert_from_path
from datetime import datetime
from process import Process
from threading import Thread
import schedule
from time import time
import datetime
from Crypto.Cipher import AES
import base64

app = Flask(__name__)


@app.route('/info', methods=['GET'])
def get():
    return 'This is Legal Document Ocr Application\nYou can contract with me by phone number: 0392200524 or email: 20130704@student.hust.edu.vn'


@app.route('/process', methods=['POST'])
def processUrl():
    if request.method == 'POST':
        if pause == False:
            filename = None
            img_name = None
            try:
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
                return process(img_name)
            except:
                return 'Error'
            finally:
                if os.path.exists(filename):
                    os.remove(filename)
                if os.path.exists(img_name):
                    os.remove(img_name)
        else:
            return('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
                văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')


@app.route('/active', methods=['POST'])
def active():
    if request.method == 'POST':
        key = request.get_json()['key']
        secret_key = '123456789012345a'
        cipher = AES.new(secret_key, AES.MODE_ECB)
        decoded = cipher.decrypt(base64.b64decode(key))
        decoded = decoded.decode('utf-8').split(' ')
        code = []
        for e in decoded:
            code.append('{}\n'.format(e))
        file = open('lcs.txt', 'r')

        ls = file.readlines()

        if ls[0] != code[0]:
            return 'Key này không có hiệu lực xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn'
        else:
            file.close()
            os.remove('lcs.txt')
            file = open('lcs.txt', 'a')
            file.writelines(code)
            file.close()
        return decoded


def time2float(t):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds = (t - epoch).total_seconds()
    return total_seconds


def job():
    if time() < time2float(datetime.datetime(2021, 4, 12)):
        return
    print('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
    văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
        sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')
    global pause
    pause = True
    print('aaa')


def licence():
    schedule.every().day.at('16:02:55').do(job)
    while True:
        schedule.run_pending()


if __name__ == '__main__':
    pause = False
    lcs = Thread(target=licence)
    lcs.start()
    process = Process('cuda')
    app.run(host='0.0.0.0', port=5000)
