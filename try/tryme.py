# try.py
from flask import Flask, Response
import cv2

app = Flask(__name__)

# 打开默认摄像头，0 表示第一个 USB 摄像头
cap = cv2.VideoCapture(0)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 编码为 JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/camera')
def camera():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Camera Stream</h1><img src="/camera">'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)