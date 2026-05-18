import socket
import struct
import time
import cv2
import numpy as np

SERVER_IP = '0.0.0.0'
SERVER_PORT = 12345

# 生成基础测试图像
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(test_img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((SERVER_IP, SERVER_PORT))
s.listen()
print(f"服务端启动，等待连接 {SERVER_IP}:{SERVER_PORT} ...")
conn, addr = s.accept()
print(f"客户端已连接：{addr}")

frame_count = 0
last_time = time.time()

while True:
    # 复制图像并添加动态时间戳
    img = test_img.copy()
    cv2.putText(img, time.strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        print("编码失败")
        continue
    data = buf.tobytes()
    conn.sendall(struct.pack('>I', len(data)))
    conn.sendall(data)
    frame_count += 1
    now = time.time()
    if now - last_time >= 1.0:
        print(f"发送帧率：{frame_count} fps")
        frame_count = 0
        last_time = now
    time.sleep(0.05)   # 约 20 fps