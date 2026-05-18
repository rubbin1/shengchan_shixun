import sys
import os
import threading
import time

sys.path.append("./MvImport")
from MvCameraControl_class import *
from CamOperation_class import CameraOperation
import web_stream

def main():
    # 1. 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail, ret =", ret)
        return
    if deviceList.nDeviceNum == 0:
        print("no device found")
        return
    print(f"Find {deviceList.nDeviceNum} devices")

    # 2. 选择第一个设备（可修改为让 C# 选择，这里先固定用索引0）
    nSelCamIndex = 0
    obj_cam = MvCamera()
    # 创建 CameraOperation 实例
    cam_op = CameraOperation(obj_cam, deviceList, nSelCamIndex)

    # 3. 打开设备
    ret = cam_op.Open_device()
    if ret != 0:
        print("open device failed")
        return
    print("device opened")

    # 4. 设置连续采集模式（可根据需要改为触发模式）
    cam_op.Set_trigger_mode("continuous")

    # 5. 将 camera_op 传给 web_stream 模块
    web_stream.camera_op = cam_op

    # 6. 启动取流和检测线程（无 GUI 版本）
    ret = cam_op.Start_grabbing_no_gui()
    if ret != 0:
        print("start grabbing failed")
        return
    print("grabbing started")

    # 7. 启动 Flask 服务（在独立线程中运行，避免阻塞）
    def run_flask():
        web_stream.start_stream(host='0.0.0.0', port=8888)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask server started on http://0.0.0.0:8888")

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping...")
        cam_op.Stop_grabbing()
        cam_op.Close_device()


if __name__ == "__main__":
    main()