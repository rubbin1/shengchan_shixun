# -*- coding: utf-8 -*-
"""
Web 视频流服务 + 相机控制 API
供 C# 上位机调用
"""

from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import threading

app = Flask(__name__)

# ---------- 全局变量，由外部 camera_service.py 设置 ----------
camera_op = None          # CameraOperation 实例
output_frame = None       # 当前检测结果图 (BGR numpy)
lock = threading.Lock()   # 保护 output_frame 的线程锁

# ---------- 视频流生成器 ----------
def generate():
    """生成 MJPEG 视频流"""
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                # 无图像时显示等待画面
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "WAITING FOR CAMERA...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
            else:
                ret, jpeg = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/camera')
def camera():
    """视频流端点"""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- 相机控制 API ----------
@app.route('/cmd/set_exposure', methods=['POST'])
def set_exposure():
    """设置相机曝光时间（单位：微秒）"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    exp = data.get('exposure_time')
    if exp is None:
        return jsonify({"error": "Missing exposure_time"}), 400

    ret, msg = camera_op.set_exposure(exp)
    if ret == 0:
        return jsonify({"status": "ok", "exposure_time": exp})
    else:
        return jsonify({"status": "fail", "code": ret, "msg": msg}), 500


@app.route('/cmd/get_exposure', methods=['GET'])
def get_exposure():
    """获取当前曝光时间"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    return jsonify({"exposure_time": camera_op.exposure_time})


@app.route('/cmd/set_gain', methods=['POST'])
def set_gain():
    """设置增益"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    gain_val = data.get('gain')
    if gain_val is None:
        return jsonify({"error": "Missing gain"}), 400

    ret, msg = camera_op.set_gain(gain_val)
    if ret == 0:
        return jsonify({"status": "ok", "gain": gain_val})
    else:
        return jsonify({"status": "fail", "code": ret, "msg": msg}), 500


@app.route('/cmd/set_frame_rate', methods=['POST'])
def set_frame_rate():
    """设置帧率"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    fps = data.get('frame_rate')
    if fps is None:
        return jsonify({"error": "Missing frame_rate"}), 400

    ret, msg = camera_op.set_frame_rate(fps)
    if ret == 0:
        return jsonify({"status": "ok", "frame_rate": fps})
    else:
        return jsonify({"status": "fail", "code": ret, "msg": msg}), 500


@app.route('/cmd/set_trigger_mode', methods=['POST'])
def set_trigger_mode():
    """设置触发模式：continuous 或 triggermode"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    mode = data.get('mode')
    if mode not in ('continuous', 'triggermode'):
        return jsonify({"error": "mode must be 'continuous' or 'triggermode'"}), 400

    camera_op.Set_trigger_mode(mode)
    return jsonify({"status": "ok", "mode": mode})


@app.route('/cmd/trigger_once', methods=['POST'])
def trigger_once():
    """软触发一次（仅在触发模式下有效）"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    camera_op.Trigger_once(1)
    return jsonify({"status": "ok"})


@app.route('/cmd/save_jpg', methods=['POST'])
def save_jpg():
    """保存当前帧为 JPG"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    camera_op.b_save_jpg = True
    return jsonify({"status": "ok"})


@app.route('/cmd/save_bmp', methods=['POST'])
def save_bmp():
    """保存当前帧为 BMP"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    if not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400

    camera_op.b_save_bmp = True
    return jsonify({"status": "ok"})


@app.route('/')
def index():
    """简易首页，提供视频流查看"""
    return '''
    <html>
    <head><title>相机实时检测</title></head>
    <body>
        <h2>零件识别实时画面</h2>
        <img src="/camera" style="max-width:100%;">
        <p>API 端点: /cmd/... 见文档</p>
    </body>
    </html>
    '''

@app.route('/cmd/enum_devices', methods=['GET'])
def enum_devices():
    """枚举相机设备，返回设备列表"""
    global camera_op
    if camera_op is None:
        return jsonify({"error": "camera_op not initialized"}), 500
    # 调用 CameraOperation 中的 get_device_list 方法
    try:
        devices = camera_op.get_device_list()
        return jsonify({"devices": devices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- 启动函数 ----------
def start_stream(host='0.0.0.0', port=8888):
    """
    启动 Flask 服务器
    一般由 camera_service.py 调用，并先设置好 camera_op 全局变量
    """
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    # 直接运行此文件仅启动空服务（无相机操作）
    print("Warning: camera_op not set. Only video stream will show WAITING...")
    start_stream()
