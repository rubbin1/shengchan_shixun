# -- coding: utf-8 --
import sys
import threading
import msvcrt
import tkinter.messagebox
from tkinter import *
from tkinter.messagebox import *
import tkinter as tk
import numpy as np
import cv2 as cv
import time
import os
import datetime
import inspect
import ctypes
import random
from PIL import Image, ImageTk
from ctypes import *          # 移到模块顶部，允许 import *
from tkinter import ttk

sys.path.append("./MvImport")
from MvCameraControl_class import *

from part_detector import PartDetector


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


class CameraOperation():

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, b_thread_closed=False, st_frame_info=None, b_exit=False,
                 b_save_bmp=False, b_save_jpg=False, buf_save_image=None,
                 n_save_image_size=0, n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0, save_path=None):
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

    def Open_device(self):
        if False == self.b_open_device:
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                tkinter.messagebox.showerror('show error', 'create handle fail! ret = ' + self.To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'open device fail! ret = ' + self.To_hex_str(ret))
                return ret
            print("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    # ------------------------------------------------------------------
    # 无 GUI 版（用于后台服务）
    # ------------------------------------------------------------------
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
        import cv2 as cv
        import numpy as np

        stOutFrame = MV_FRAME_OUT()
        img_buff = None  # 用于存储转换后的RGB数据
        buf_cache = None
        DISPLAY_W, DISPLAY_H = 800, 600
        self.frame_count = 0
        self.last_detected = None

        while not self.b_exit:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret != 0:
                continue

            if buf_cache is None:
                buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
            self.st_frame_info = stOutFrame.stFrameInfo

            # 确保 img_buff 足够大
            nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
            if img_buff is None or len(img_buff) < nConvertSize:
                img_buff = (c_ubyte * nConvertSize)()

            # 像素转换
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
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                if ret != 0:
                    print("convert pixel fail, ret =", self.To_hex_str(ret))
                    self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                    continue
                numArray = self.Color_numpy(img_buff, self.st_frame_info.nWidth, self.st_frame_info.nHeight)

            # 转为 BGR 并缩放
            bgr_big = cv.cvtColor(numArray, cv.COLOR_RGB2BGR)
            bgr_small = cv.resize(bgr_big, (DISPLAY_W, DISPLAY_H))

            # 零件检测（跳帧）
            self.frame_count += 1
            if self.frame_count % 3 == 0 or self.last_detected is None:
                detected_small = self.detector.detect(bgr_small)
                self.last_detected = detected_small.copy()
            else:
                detected_small = self.last_detected

            # 更新全局帧
            if detected_small is not None:
                with web_stream.lock:
                    web_stream.output_frame = detected_small.copy()

            self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)

        print("Work_thread_no_gui exited")

    # ------------------------------------------------------------------
    # 原有带 Tkinter 的版本（保留完整）
    # ------------------------------------------------------------------
    def Start_grabbing(self, root, panel):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'start grabbing fail! ret = ' + self.To_hex_str(ret))
                return
            self.b_start_grabbing = True
            print("start grabbing successfully!")
            try:
                self.n_win_gui_id = random.randint(1, 10000)
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread, args=(self, root, panel))
                self.h_thread_handle.start()
                self.b_thread_closed = True
            except:
                tkinter.messagebox.showerror('show error', 'error: unable to start thread')
                False == self.b_start_grabbing

    def Work_thread(self, root, panel):
        stOutFrame = MV_FRAME_OUT()
        img_buff = None
        buf_cache = None
        numArray = None

        DISPLAY_W, DISPLAY_H = 800, 600
        self.frame_count = 0
        self.last_detected = None

        while not self.b_exit:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                if None == buf_cache:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()

                if True == self.b_save_jpg:
                    self.Save_jpg(buf_cache)
                if True == self.b_save_bmp:
                    self.Save_Bmp(buf_cache)
            else:
                print("no data, nret = " + self.To_hex_str(ret))
                continue

            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType

            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                numArray = CameraOperation.Color_numpy(self, buf_cache,
                                                       self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
            else:
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = CameraOperation.Color_numpy(self, img_buff,
                                                       self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)

            bgr_big = cv.cvtColor(numArray, cv.COLOR_RGB2BGR)
            bgr_small = cv.resize(bgr_big, (DISPLAY_W, DISPLAY_H))

            self.frame_count += 1
            if self.frame_count % 3 == 0 or self.last_detected is None:
                detected_small = self.detector.detect(bgr_small)
                self.last_detected = detected_small.copy()
            else:
                detected_small = self.last_detected

            detected_rgb = cv.cvtColor(detected_small, cv.COLOR_BGR2RGB)

            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.ANTIALIAS
            current_image = Image.fromarray(detected_rgb)
            imgtk = ImageTk.PhotoImage(image=current_image, master=root)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            root.obr = imgtk

            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            if self.b_exit:
                if img_buff is not None:
                    del img_buff
                if buf_cache is not None:
                    del buf_cache
                break

    # ------------------------------------------------------------------
    # 停止和关闭（使用 b_exit 优雅退出）
    # ------------------------------------------------------------------
    def Stop_grabbing(self):
        if self.b_start_grabbing and self.b_open_device:
            self.b_exit = True
            if self.h_thread_handle and self.h_thread_handle.is_alive():
                self.h_thread_handle.join(timeout=2)
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'stop grabbing fail! ret = ' + self.To_hex_str(ret))
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
                tkinter.messagebox.showerror('show error', 'close deivce fail! ret = ' + self.To_hex_str(ret))
                return
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        print("close device successfully!")

    # ------------------------------------------------------------------
    # 参数设置单独方法（供 API 调用）
    # ------------------------------------------------------------------
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
            # 先使能帧率控制
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
        if True == self.b_open_device:
            if "continuous" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 0)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggermode fail! ret = ' + self.To_hex_str(ret))
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 1)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggermode fail! ret = ' + self.To_hex_str(ret))
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggersource fail! ret = ' + self.To_hex_str(ret))

    def Trigger_once(self, nCommand):
        if True == self.b_open_device:
            if 1 == nCommand:
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggersoftware fail! ret = ' + self.To_hex_str(ret))

    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'get acquistion frame rate fail! ret = ' + self.To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'get exposure time fail! ret = ' + self.To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'get gain fail! ret = ' + self.To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            tkinter.messagebox.showinfo('show info', 'get parameter success!')

    def get_device_list(self):
        """
        返回所有枚举到的设备字符串列表，格式如:
        ["[0]GigE: 相机名(192.168.1.100)", "[1]USB: 相机名(SN12345)"]
        """
        dev_list = []
        for i in range(self.st_device_list.nDeviceNum):
            mvcc_dev_info = cast(self.st_device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                # 解析 GigE 设备名称和 IP
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
                # 解析 USB 设备名称和序列号
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

    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            tkinter.messagebox.showinfo('show info', 'please type in the text box !')
            return
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'set exposure time fail! ret = ' + self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'set gain fail! ret = ' + self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'set acquistion frame rate fail! ret = ' + self.To_hex_str(ret))

            tkinter.messagebox.showinfo('show info', 'set parameter success!')

    # ------------------------------------------------------------------
    # 保存图像
    # ------------------------------------------------------------------
    def Save_jpg(self, buf_cache):
        if (None == buf_cache):
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
            tkinter.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            tkinter.messagebox.showinfo('show info', 'save jpg success!')
        except:
            self.b_save_jpg = False
            raise Exception("save jpg failed")
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Save_Bmp(self, buf_cache):
        if (0 == buf_cache):
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
            tkinter.messagebox.showerror('show error', 'save bmp fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tkinter.messagebox.showinfo('show info', 'save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("save bmp failed")
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    # ------------------------------------------------------------------
    # 像素格式辅助函数
    # ------------------------------------------------------------------
    def Is_mono_data(self, enGvspPixelType):
        if PixelType_Gvsp_Mono8 == enGvspPixelType or PixelType_Gvsp_Mono10 == enGvspPixelType \
                or PixelType_Gvsp_Mono10_Packed == enGvspPixelType or PixelType_Gvsp_Mono12 == enGvspPixelType \
                or PixelType_Gvsp_Mono12_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Is_color_data(self, enGvspPixelType):
        if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
                or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
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