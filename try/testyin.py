from flask import Flask, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# 生成基础测试图像（保留你们原有的逻辑）
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(test_img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def generate_frames():
    """生成视频帧（无限循环，每次 yield 一帧 JPEG）"""
    while True:
        # 复制图像并添加动态时间戳
        img = test_img.copy()
        cv2.putText(img, time.strftime("%H:%M:%S"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 编码为 JPEG
        ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        # 拼接成 MJPEG 流的一帧（格式：--boundary\r\nContent-Type...\r\n\r\n图像数据）
        frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        yield frame

        # 控制帧率（约 20 fps）
        time.sleep(0.05)

@app.route('/camera')
def camera():
    """MJPEG 视频流地址"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("MJPEG 流服务启动，访问 http://<本机IP>:8888/camera")
    # 监听所有网卡，端口 8080，允许局域网内其他设备访问
    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)