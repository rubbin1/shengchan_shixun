# -*- coding: utf-8 -*-
"""
Web 视频流服务 + 相机控制 API（无 GUI 版）
供 C# 上位机调用
"""

from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import threading
import sys
import os

# 把 SDK 路径加入 sys.path（确保和主程序一致）
sys.path.append("./MvImport")
from MvCameraControl_class import *
from CamOperation_class import CameraOperation

app = Flask(__name__)

# ---------- 全局变量 ----------
camera_op = None              # CameraOperation 实例
output_frame = None           # 当前检测结果图 (BGR numpy)
lock = threading.Lock()       # 保护 output_frame 和 detected_parts

cached_device_list = []       # 启动后缓存的设备列表（字符串）
deviceList_raw = None         # 原始 SDK 设备列表对象，用于打开相机

detected_parts = []           # 当前帧的零件列表，由取流线程更新


# ---------- 视频流生成器 ----------
def generate():
    """MJPEG 视频流，当无图像时显示等待画面"""
    global output_frame
    while True:
        with lock:
            if output_frame is None:
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
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- 相机控制 API ----------
@app.route('/cmd/enum_devices', methods=['GET'])
def enum_devices():
    """返回缓存的设备列表"""
    global cached_device_list
    return jsonify({"devices": cached_device_list})


@app.route('/cmd/open_device', methods=['POST'])
def open_device():
    """
    打开指定索引的相机，并开始取流
    JSON 参数: {"device_index": 0}
    """
    global camera_op, deviceList_raw, output_frame
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    idx = data.get('device_index', 0)

    # 如果已经打开了相机，先关闭
    if camera_op is not None and camera_op.b_open_device:
        camera_op.Stop_grabbing()
        camera_op.Close_device()
        camera_op = None

    # 检查设备列表是否有效
    if deviceList_raw is None or idx >= deviceList_raw.nDeviceNum:
        return jsonify({"error": "Invalid device index or no device enumerated"}), 400

    # 创建 CameraOperation 实例
    obj_cam = MvCamera()
    cam_op = CameraOperation(obj_cam, deviceList_raw, idx)
    ret = cam_op.Open_device()
    if ret != 0:
        return jsonify({"error": f"Open_device failed, ret={hex(ret)}"}), 500

    # 设置连续采集模式
    cam_op.Set_trigger_mode("continuous")

    # 开始取流（内部线程会自动更新 web_stream.output_frame 和 web_stream.detected_parts）
    ret = cam_op.Start_grabbing_no_gui()
    if ret != 0:
        return jsonify({"error": "Start_grabbing_no_gui failed"}), 500

    camera_op = cam_op
    return jsonify({"status": "ok", "device_index": idx})


@app.route('/cmd/close_device', methods=['POST'])
def close_device():
    """关闭当前相机"""
    global camera_op, output_frame
    if camera_op is None:
        return jsonify({"status": "already closed"})
    camera_op.Stop_grabbing()
    camera_op.Close_device()
    camera_op = None
    with lock:
        output_frame = None
    return jsonify({"status": "ok"})


# ---------- 零件检测结果 API ----------
@app.route('/cmd/detect_result', methods=['GET'])
def detect_result():
    """返回最近一帧的零件检测结果"""
    global detected_parts
    with lock:
        parts_copy = list(detected_parts) if detected_parts else []
    return jsonify({"parts": parts_copy})


# ---------- 参数设置 API ----------
@app.route('/cmd/set_exposure', methods=['POST'])
def set_exposure():
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    data = request.get_json()
    exp = data.get('exposure_time')
    if exp is None:
        return jsonify({"error": "Missing exposure_time"}), 400
    ret, msg = camera_op.set_exposure(exp)
    if ret == 0:
        return jsonify({"status": "ok", "exposure_time": exp})
    else:
        return jsonify({"status": "fail", "code": ret, "msg": msg}), 500


@app.route('/cmd/set_gain', methods=['POST'])
def set_gain():
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    data = request.get_json()
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
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    data = request.get_json()
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
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    data = request.get_json()
    mode = data.get('mode')
    if mode not in ('continuous', 'triggermode'):
        return jsonify({"error": "mode must be 'continuous' or 'triggermode'"}), 400
    camera_op.Set_trigger_mode(mode)
    return jsonify({"status": "ok", "mode": mode})


@app.route('/cmd/trigger_once', methods=['POST'])
def trigger_once():
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    camera_op.Trigger_once(1)
    return jsonify({"status": "ok"})


@app.route('/cmd/save_jpg', methods=['POST'])
def save_jpg():
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    camera_op.b_save_jpg = True
    return jsonify({"status": "ok"})


@app.route('/cmd/save_bmp', methods=['POST'])
def save_bmp():
    global camera_op
    if camera_op is None or not camera_op.b_open_device:
        return jsonify({"error": "camera not opened"}), 400
    camera_op.b_save_bmp = True
    return jsonify({"status": "ok"})


@app.route('/')
def index():
    return '''
    <html>
    <head><title>相机实时检测</title></head>
    <body>
        <h2>零件识别实时画面</h2>
        < img src="/camera" style="max-width:100%;">
        <p>API 端点: /cmd/... 见文档</p >
    </body>
    </html>
    '''


# ---------- 启动函数 ----------
def start_stream(host='0.0.0.0', port=8888):
    app.run(host=host, port=port, threaded=True, debug=False)