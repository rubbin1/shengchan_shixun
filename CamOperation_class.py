# -*- coding: utf-8 -*-
import sys
import threading
import os
import ctypes
import inspect
import numpy as np
import cv2 as cv
from ctypes import *

sys.path.append("./MvImport")
from MvCameraControl_class import *
from part_detector import PartDetector

# ---------- 线程强制退出工具（保留） ----------
def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


class CameraOperation:
    def __init__(self, obj_cam, st_device_list, n_connect_num=0,
                 b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, b_thread_closed=False,
                 st_frame_info=None, b_exit=False,
                 b_save_bmp=False, b_save_jpg=False,
                 buf_save_image=None, n_save_image_size=0,
                 n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0,
                 save_path=None):
        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.detector = PartDetector(min_area_ratio=0.0005, debug=False)

        if save_path is None:
            self.save_path = os.getcwd()
        else:
            self.save_path = save_path

    # -----------------------------------------------------------------
    # 辅助：数字转十六进制字符串
    # -----------------------------------------------------------------
    def To_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2 ** 32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    # -----------------------------------------------------------------
    # 打开设备（无 GUI）
    # -----------------------------------------------------------------
    def Open_device(self):
        if self.b_open_device:
            return 0
        nConnectionNum = int(self.n_connect_num)
        stDeviceList = cast(self.st_device_list.pDeviceInfo[nConnectionNum],
                            POINTER(MV_CC_DEVICE_INFO)).contents
        self.obj_cam = MvCamera()
        ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            self.obj_cam.MV_CC_DestroyHandle()
            print(f"create handle fail! ret = {self.To_hex_str(ret)}")
            return ret

        ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"open device fail! ret = {self.To_hex_str(ret)}")
            return ret
        print("open device successfully!")
        self.b_open_device = True
        self.b_thread_closed = False

        # GigE 优化
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print(f"warning: set packet size fail! ret[0x{ret:x}]")
            else:
                print(f"warning: get optimal packet size fail! ret[0x{nPacketSize:x}]")

        # 默认关闭帧率控制（后续可单独打开）
        self.obj_cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
        # 关闭触发模式
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"set trigger mode off fail! ret[0x{ret:x}]")
        return 0

    # -----------------------------------------------------------------
    # 开始取流（无 GUI 版本）
    # -----------------------------------------------------------------
    def Start_grabbing_no_gui(self):
        if self.b_start_grabbing:
            return 0
        if not self.b_open_device:
            return -1
        self.b_exit = False
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            return ret
        self.b_start_grabbing = True
        self.h_thread_handle = threading.Thread(target=self.Work_thread_no_gui, daemon=True)
        self.h_thread_handle.start()
        self.b_thread_closed = True
        return 0

    def Work_thread_no_gui(self):
        import web_stream
        import traceback
        import time

        stOutFrame = MV_FRAME_OUT()
        img_buff = None
        buf_cache = None
        DISPLAY_W, DISPLAY_H = 800, 600
        self.frame_count = 0
        self.last_detected = None

        while not self.b_exit:
            try:
                # ---- 1. 获取一帧图像 ----
                ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
                if ret != 0:
                    continue

                if buf_cache is None:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                self.st_frame_info = stOutFrame.stFrameInfo

                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                if img_buff is None or len(img_buff) < nConvertSize:
                    img_buff = (c_ubyte * nConvertSize)()

                # ---- 2. 像素格式转换为 RGB 数组 ----
                if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                    numArray = self.Color_numpy(buf_cache, self.st_frame_info.nWidth, self.st_frame_info.nHeight)
                else:
                    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                    stConvertParam.nWidth = self.st_frame_info.nWidth
                    stConvertParam.nHeight = self.st_frame_info.nHeight
                    stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
                    stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
                    stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType
                    stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                    stConvertParam.pDstBuffer = img_buff
                    stConvertParam.nDstBufferSize = nConvertSize
                    ret_conv = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                    if ret_conv != 0:
                        print(f"[Warning] convert pixel fail, ret = {self.To_hex_str(ret_conv)}")
                        self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                        continue
                    numArray = self.Color_numpy(img_buff, self.st_frame_info.nWidth, self.st_frame_info.nHeight)

                # ---- 3. 转 BGR 并缩放到显示尺寸 ----
                bgr_big = cv.cvtColor(numArray, cv.COLOR_RGB2BGR)
                bgr_small = cv.resize(bgr_big, (DISPLAY_W, DISPLAY_H))

                # ---- 4. 零件检测（跳帧 + 异常隔离） ----
                self.frame_count += 1
                detected_small = bgr_small.copy()  # 默认使用原始图像
                try:
                    # 跳帧检测：每3帧检测一次，或首次检测
                    if self.frame_count % 3 == 0 or self.last_detected is None:
                        detected_small = self.detector.detect(bgr_small)
                        self.last_detected = detected_small.copy()
                    else:
                        detected_small = self.last_detected
                        # 如果 last_detected 意外为 None，退回到原始图像
                        if detected_small is None:
                            detected_small = bgr_small.copy()
                except Exception as e:
                    print(f"[Warning] PartDetector error: {e}")
                    traceback.print_exc()
                    detected_small = bgr_small.copy()
                    self.last_detected = bgr_small.copy()

                # ---- 4.5 提取零件信息并更新全局变量 ----
                try:
                    if hasattr(self, 'detector') and hasattr(self.detector, 'last_parts'):
                        parts_info = []
                        for part in self.detector.last_parts:
                            if isinstance(part, dict):
                                part_copy = {
                                    "id": part["id"],
                                    "center": part["center"],
                                    "area": part["area"],
                                    "contour": part["contour"].tolist() if isinstance(part["contour"], np.ndarray) else
                                    part["contour"],
                                    "color": part.get("color", ""),
                                    "shape": part.get("shape", "")
                                }
                                parts_info.append(part_copy)
                        with web_stream.lock:
                            web_stream.detected_parts = parts_info
                    else:
                        with web_stream.lock:
                            web_stream.detected_parts = []
                except Exception as e:
                    print(f"[Warning] Failed to extract parts info: {e}")
                    with web_stream.lock:
                        web_stream.detected_parts = []

                # ---- 5. 更新全局帧 ----
                if detected_small is not None:
                    with web_stream.lock:
                        web_stream.output_frame = detected_small.copy()

                # ---- 6. 释放 SDK 图像缓存 ----
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)

            except Exception as e:
                print(f"[ERROR] Work_thread_no_gui inner loop error: {e}")
                traceback.print_exc()
                try:
                    self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                except:
                    pass
                time.sleep(0.1)

        print("Work_thread_no_gui exited")

    # -----------------------------------------------------------------
    # 停止取流与关闭设备（无 GUI）
    # -----------------------------------------------------------------
    def Stop_grabbing(self):
        if self.b_start_grabbing and self.b_open_device:
            self.b_exit = True
            if self.h_thread_handle and self.h_thread_handle.is_alive():
                self.h_thread_handle.join(timeout=2)
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                print(f"stop grabbing fail! ret = {self.To_hex_str(ret)}")
                return
            print("stop grabbing successfully!")
            self.b_start_grabbing = False

    def Close_device(self):
        if self.b_open_device:
            self.b_exit = True
            if self.h_thread_handle and self.h_thread_handle.is_alive():
                self.h_thread_handle.join(timeout=2)
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                print(f"close device fail! ret = {self.To_hex_str(ret)}")
                return
            self.obj_cam.MV_CC_DestroyHandle()
            self.b_open_device = False
            self.b_start_grabbing = False
            print("close device successfully!")

    # -----------------------------------------------------------------
    # 参数设置（供 Flask API 调用）
    # -----------------------------------------------------------------
    def set_exposure(self, exposure_time_us):
        if not self.b_open_device:
            return -1, "Camera not opened"
        try:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time_us))
            if ret == 0:
                self.exposure_time = exposure_time_us
                return 0, "OK"
            else:
                return ret, f"SetFloatValue failed with code {self.To_hex_str(ret)}"
        except Exception as e:
            return -2, str(e)

    def set_gain(self, gain_val):
        if not self.b_open_device:
            return -1, "Camera not opened"
        try:
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain_val))
            if ret == 0:
                self.gain = gain_val
                return 0, "OK"
            else:
                return ret, f"SetFloatValue failed with code {self.To_hex_str(ret)}"
        except Exception as e:
            return -2, str(e)

    def set_frame_rate(self, fps):
        if not self.b_open_device:
            return -1, "Camera not opened"
        try:
            # 使能帧率控制
            self.obj_cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
            if ret == 0:
                self.frame_rate = fps
                return 0, "OK"
            else:
                return ret, f"SetFloatValue failed with code {self.To_hex_str(ret)}"
        except Exception as e:
            return -2, str(e)

    def Set_trigger_mode(self, strMode):
        if not self.b_open_device:
            return
        if strMode == "continuous":
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 0)
            if ret != 0:
                print(f"set trigger mode continuous fail! ret = {self.To_hex_str(ret)}")
        elif strMode == "triggermode":
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 1)
            if ret != 0:
                print(f"set trigger mode on fail! ret = {self.To_hex_str(ret)}")
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7)  # software
            if ret != 0:
                print(f"set trigger source software fail! ret = {self.To_hex_str(ret)}")
        else:
            print(f"unknown trigger mode: {strMode}")

    def Trigger_once(self, nCommand):
        if self.b_open_device and nCommand == 1:
            ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
            if ret != 0:
                print(f"trigger software fail! ret = {self.To_hex_str(ret)}")

    # -----------------------------------------------------------------
    # 获取设备列表（字符串）
    # -----------------------------------------------------------------
    def get_device_list(self):
        dev_list = []
        for i in range(self.st_device_list.nDeviceNum):
            mvcc_dev_info = cast(self.st_device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName += chr(per)
                ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
                nip1 = (ip >> 24) & 0xFF
                nip2 = (ip >> 16) & 0xFF
                nip3 = (ip >> 8) & 0xFF
                nip4 = ip & 0xFF
                dev_list.append(f"[{i}]GigE: {chUserDefinedName}({nip1}.{nip2}.{nip3}.{nip4})")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName += chr(per)
                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                dev_list.append(f"[{i}]USB: {chUserDefinedName}({strSerialNumber})")
        return dev_list

    # -----------------------------------------------------------------
    # 保存图像（内部调用，无弹窗）
    # -----------------------------------------------------------------
    def Save_jpg(self, buf_cache):
        if buf_cache is None:
            return
        self.buf_save_image = None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, str(self.st_frame_info.nFrameNum) + ".jpg")
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg
        stParam.enPixelType = self.st_frame_info.enPixelType
        stParam.nWidth = self.st_frame_info.nWidth
        stParam.nHeight = self.st_frame_info.nHeight
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size
        stParam.nJpgQuality = 80
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            print(f"save jpg fail! ret = {self.To_hex_str(return_code)}")
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            print(f"save jpg success: {file_path}")
        except Exception as e:
            self.b_save_jpg = False
            print(f"save jpg failed: {e}")
        finally:
            file_open.close()
            del img_buff
            del self.buf_save_image

    def Save_Bmp(self, buf_cache):
        if buf_cache is None:
            return
        self.buf_save_image = None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, str(self.st_frame_info.nFrameNum) + ".bmp")
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp
        stParam.enPixelType = self.st_frame_info.enPixelType
        stParam.nWidth = self.st_frame_info.nWidth
        stParam.nHeight = self.st_frame_info.nHeight
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)
        if return_code != 0:
            print(f"save bmp fail! ret = {self.To_hex_str(return_code)}")
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            print(f"save bmp success: {file_path}")
        except Exception as e:
            self.b_save_bmp = False
            print(f"save bmp failed: {e}")
        finally:
            file_open.close()
            del img_buff
            del self.buf_save_image

    # -----------------------------------------------------------------
    # 像素格式判断与转换
    # -----------------------------------------------------------------
    def Is_mono_data(self, enGvspPixelType):
        return enGvspPixelType in (PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10,
                                   PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12,
                                   PixelType_Gvsp_Mono12_Packed)

    def Is_color_data(self, enGvspPixelType):
        return enGvspPixelType in (
            PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10,
            PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerRG12,
            PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12,
            PixelType_Gvsp_BayerGR10_Packed, PixelType_Gvsp_BayerRG10_Packed,
            PixelType_Gvsp_BayerGB10_Packed, PixelType_Gvsp_BayerBG10_Packed,
            PixelType_Gvsp_BayerGR12_Packed, PixelType_Gvsp_BayerRG12_Packed,
            PixelType_Gvsp_BayerGB12_Packed, PixelType_Gvsp_BayerBG12_Packed,
            PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed
        )

    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=nWidth * nHeight, dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=nWidth * nHeight * 3, dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]
        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")
        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray