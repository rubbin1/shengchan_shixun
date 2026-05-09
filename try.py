import cv2 as cv
import numpy as np

# 1. 读取图像
img = cv.imread("pictures/spare parts/yellow_liubian.jpg")
if img is None:
    print("图片未找到")
    exit()

# 2. 转换到 HSV 颜色空间（便于提取特定颜色）
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 3. 定义橙色的 HSV 范围（可能需要根据实际光照微调）
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])

# 4. 创建掩膜，只保留黄色区域
mask = cv.inRange(hsv, lower_orange, upper_orange)

# 5. 形态学操作（去除小的噪点和填充内部空洞）
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # 闭运算：先膨胀后腐蚀，填补小黑洞
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # 开运算：先腐蚀后膨胀，去掉小白点

# 6. 查找轮廓
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 7. 遍历轮廓，筛选六边形并绘制
for cnt in contours:
    # 计算轮廓周长
    perimeter = cv.arcLength(cnt, True)
    # 多边形逼近（得到顶点更少的近似轮廓）
    approx = cv.approxPolyDP(cnt, 0.03 * perimeter, True)

    # 如果是六边形（6个顶点），且轮廓面积足够大，避免噪声
    if len(approx) == 6 and cv.contourArea(cnt) > 500:
        # 在原图上绘制轮廓
        cv.drawContours(img, [approx], -1, (0, 255, 0), 3)  # 绿色加粗轮廓
        # 也可以绘制最小外接矩形或顶点标记
        # for point in approx:
        #     cv.circle(img, tuple(point[0]), 5, (0,0,255), -1)

# 8. 显示结果（窗口可调整大小）
cv.namedWindow("Detected Yellow Hexagon", cv.WINDOW_NORMAL)
cv.imshow("Detected Yellow Hexagon", img)
cv.waitKey(0)
cv.destroyAllWindows()