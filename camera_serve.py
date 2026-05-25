import sys
import os
import threading
import time

sys.path.append("./MvImport")
from MvCameraControl_class import *
from CamOperation_class import CameraOperation
import web_stream

def main():
    # ---------- 1. 枚举设备 ----------
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail, ret =", ret)
        return
    if deviceList.nDeviceNum == 0:
        print("no device found")
    else:
        print(f"Find {deviceList.nDeviceNum} devices")

    # 缓存到 web_stream
    web_stream.deviceList_raw = deviceList
    if deviceList.nDeviceNum > 0:
        # 构造字符串列表用于 C# 显示（临时对象即可，因为 get_device_list 需要 st_device_list 存在）
        # 为方便，我们可以直接调用一次 CamOperation 的静态方法来获取列表，
        # 但这里需要 st_device_list 对象，所以先创建一个临时 cam_op 仅用于列表生成。
        temp_cam = MvCamera()
        temp_op = CameraOperation(temp_cam, deviceList, 0)
        web_stream.cached_device_list = temp_op.get_device_list()
    else:
        web_stream.cached_device_list = []

    # ---------- 2. 启动 Flask 服务 ----------
    def run_flask():
        web_stream.start_stream(host='0.0.0.0', port=8888)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask server started on http://0.0.0.0:8888")
    print("Waiting for commands from C# client...")

    # ---------- 3. 保持主线程运行 ----------
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if web_stream.camera_op is not None:
            web_stream.camera_op.Stop_grabbing()
            web_stream.camera_op.Close_device()
        print("Exit.")


if __name__ == "__main__":
    main()