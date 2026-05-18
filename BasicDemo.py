# -- coding: utf-8 --
import sys
from tkinter import *
import _tkinter
import tkinter.messagebox
import tkinter as tk
from tkinter import filedialog
import sys, os
from tkinter import ttk
sys.path.append("./MvImport")
from MvCameraControl_class import *
from CamOperation_class import *
from PIL import Image,ImageTk

import web_stream
import threading

#获取选取设备信息的索引，通过[]之间的字符去解析
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()

#将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr

if __name__ == "__main__":
    global deviceList 
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global tlayerType
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global b_is_run
    b_is_run = False

    global save_folder
    save_folder = os.getcwd()

    #界面设计代码
    window = tk.Tk()
    window.title('BasicDemo')
    window.geometry('1150x650')
    model_val = tk.StringVar()
    global triggercheck_val
    triggercheck_val = tk.IntVar()
    page = Frame(window,height=400,width=60,relief=GROOVE,bd=5,borderwidth=4)
    page.pack(expand=True, fill=BOTH)
    panel = Label(page)
    panel.place(x=190, y=10,height=600,width=1000)
        #绑定下拉列表至设备信息索引
    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[","]",device_list.get())

    #ch:枚举相机 | en:enum devices
    def enum_devices():
        global deviceList
        global obj_cam_operation
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            tkinter.messagebox.showerror('show error','enum devices fail! ret = '+ ToHexStr(ret))

        if deviceList.nDeviceNum == 0:
            tkinter.messagebox.showinfo('show info','find no device!')

        print ("Find %d devices!" % deviceList.nDeviceNum)

        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print ("device model name: %s" % chUserDefinedName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devList.append("["+str(i)+"]GigE: "+ chUserDefinedName +"("+ str(nip1)+"."+str(nip2)+"."+str(nip3)+"."+str(nip4) +")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print ("device model name: %s" % chUserDefinedName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
                devList.append("["+str(i)+"]USB: "+ chUserDefinedName +"(" + str(strSerialNumber) + ")")
        device_list["value"] = devList
        device_list.current(0)
    
        #ch:打开相机 | en:open device
    def open_device():
        global deviceList, nSelCamIndex, obj_cam_operation, b_is_run, save_folder
        if b_is_run:
            tkinter.messagebox.showinfo('show info', 'Camera is Running!')
            return
        obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if 0 != ret:
            b_is_run = False
        else:
            model_val.set('continuous')
            b_is_run = True
            # 启动Web流线程（只启动一次）
            if not hasattr(web_stream, 'thread_started'):
                web_thread = threading.Thread(target=web_stream.start_stream, daemon=True)
                web_thread.start()
                web_stream.thread_started = True
                print("Web 视频流已启动，浏览器访问 http://<本机IP>:8888/camera")
            obj_cam_operation.save_path = save_folder  # *** 新增：同步保存路径 ***

    # ch:开始取流 | en:Start grab image
    def start_grabbing():
        global obj_cam_operation
        obj_cam_operation.Start_grabbing(window,panel)

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing():
        global obj_cam_operation
        obj_cam_operation.Stop_grabbing()    

    # ch:关闭设备 | Close device   
    def close_device():
        global b_is_run
        global obj_cam_operation
        obj_cam_operation.Close_device()
        b_is_run = False 
        #清除文本框的数值
        text_frame_rate.delete(1.0, tk.END)
        text_exposure_time.delete(1.0, tk.END)
        text_gain.delete(1.0, tk.END)

    #ch:设置触发模式 | en:set trigger mode
    def set_triggermode():
        global obj_cam_operation
        strMode = model_val.get()
        obj_cam_operation.Set_trigger_mode(strMode)

    #ch:设置触发命令 | en:set trigger software
    def trigger_once():
        global triggercheck_val
        global obj_cam_operation
        nCommand = triggercheck_val.get()
        obj_cam_operation.Trigger_once(nCommand)
    
    #ch:保存bmp图片 | en:save bmp image
    def bmp_save():
        global obj_cam_operation
        obj_cam_operation.b_save_bmp = True

    #选择保存图片的文件夹
    def select_folder():
        global save_folder
        global obj_cam_operation
        path = tk.filedialog.askdirectory(title='选择图片保存文件夹')
        if path:  # 用户没有取消
            save_folder = path
            label_save_path.config(text='保存路径：' + save_folder)
            # 如果相机已打开，立即更新对象属性
            if obj_cam_operation != 0:
                obj_cam_operation.save_path = save_folder

    #ch:保存jpg图片 | en:save jpg image
    def jpg_save():
        global obj_cam_operation
        obj_cam_operation.b_save_jpg = True


    def get_parameter():
        global obj_cam_operation
        obj_cam_operation.Get_parameter()
        text_frame_rate.delete(1.0, tk.END)
        text_frame_rate.insert(1.0,obj_cam_operation.frame_rate)
        text_exposure_time.delete(1.0, tk.END)
        text_exposure_time.insert(1.0,obj_cam_operation.exposure_time)
        text_gain.delete(1.0, tk.END)
        text_gain.insert(1.0, obj_cam_operation.gain)

    def set_parameter():
        global obj_cam_operation
        obj_cam_operation.exposure_time = text_exposure_time.get(1.0,tk.END)
        obj_cam_operation.exposure_time = obj_cam_operation.exposure_time.rstrip("\n")
        obj_cam_operation.gain = text_gain.get(1.0,tk.END)
        obj_cam_operation.gain = obj_cam_operation.gain.rstrip("\n")
        obj_cam_operation.frame_rate = text_frame_rate.get(1.0,tk.END)
        obj_cam_operation.frame_rate = obj_cam_operation.frame_rate.rstrip("\n")
        obj_cam_operation.Set_parameter(obj_cam_operation.frame_rate,obj_cam_operation.exposure_time,obj_cam_operation.gain)

    xVariable = tkinter.StringVar()
    device_list = ttk.Combobox(window, textvariable=xVariable,width=30)
    device_list.place(x=20, y=20)
    device_list.bind("<<ComboboxSelected>>", xFunc)

    label_exposure_time = tk.Label(window, text='曝光时间',width=15, height=1)
    label_exposure_time.place(x=20, y=400)
    text_exposure_time = tk.Text(window,width=15, height=1)
    text_exposure_time.place(x=160, y=400)

    label_gain = tk.Label(window, text='增益', width=15, height=1)
    label_gain.place(x=20, y=450)
    text_gain = tk.Text(window,width=15, height=1)
    text_gain.place(x=160, y=450)

    label_frame_rate = tk.Label(window, text='帧率', width=15, height=1)
    label_frame_rate.place(x=20, y=500)
    text_frame_rate  = tk.Text(window,width=15, height=1)
    text_frame_rate.place(x=160, y=500)

    btn_enum_devices = tk.Button(window, text='枚举设备', width=35, height=1, command = enum_devices )
    btn_enum_devices.place(x=20, y=50)
    btn_open_device = tk.Button(window, text='打开设备', width=15, height=1, command = open_device)
    btn_open_device.place(x=20, y=100)
    btn_close_device = tk.Button(window, text='关闭设备', width=15, height=1, command = close_device)
    btn_close_device.place(x=160, y=100)

    radio_continuous = tk.Radiobutton(window, text='连续采集',variable=model_val, value='continuous',width=15, height=1,command=set_triggermode)
    radio_continuous.place(x=20,y=150)
    radio_trigger = tk.Radiobutton(window, text='触发模式',variable=model_val, value='triggermode',width=15, height=1,command=set_triggermode)
    radio_trigger.place(x=160,y=150)
    model_val.set(1)

    btn_start_grabbing = tk.Button(window, text='开始采集', width=15, height=1, command = start_grabbing )
    btn_start_grabbing.place(x=20, y=200)
    btn_stop_grabbing = tk.Button(window, text='停止采集', width=15, height=1, command = stop_grabbing)
    btn_stop_grabbing.place(x=160, y=200)

    checkbtn_trigger_software = tk.Checkbutton(window, text='软触发', variable=triggercheck_val, onvalue=1, offvalue=0)
    checkbtn_trigger_software.place(x=20,y=250)
    btn_trigger_once = tk.Button(window, text='单次触发', width=15, height=1, command = trigger_once)
    btn_trigger_once.place(x=160, y=250)

    btn_save_bmp = tk.Button(window, text='保存BMP', width=15, height=1, command = bmp_save )
    btn_save_bmp.place(x=20, y=300)
    btn_save_jpg = tk.Button(window, text='保存JPG', width=15, height=1, command = jpg_save)
    btn_save_jpg.place(x=160, y=300)

    # 显示当前保存路径
    label_save_path = tk.Label(window, text='保存路径：' + save_folder, anchor='w', width=60)
    label_save_path.place(x=20, y=330)

    # 选择文件夹按钮
    btn_select_folder = tk.Button(window, text='选择保存文件夹', width=15, height=1, command=select_folder)
    btn_select_folder.place(x=20, y=360)

    btn_get_parameter = tk.Button(window, text='获取参数', width=15, height=1, command = get_parameter)
    btn_get_parameter.place(x=20, y=550)
    btn_set_parameter = tk.Button(window, text='设置参数', width=15, height=1, command = set_parameter)
    btn_set_parameter.place(x=160, y=550)
    window.mainloop()

