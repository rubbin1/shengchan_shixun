import cv2 as cv
import numpy as np

#橙色六边形：lower: hsv: 0 165 155
#           upper: hsv: 30 255 255
#浅橙色三角形：lower: hsv: 0 80 90
#           upper: hsv: 20 162 155
#粉色菱形：lower: hsv: 150 85 80
#        upper: hsv: 175 255 255
#绿色梯形：lower: hsv: 35 50 50
#        upper: hsv: 90 255 255

class Part:
    def __init__(self, color: str, shape: str):
        self.color = color
        self.shape = shape

#定义四种颜色和形状
color_types = {
    "绿色": ([35, 50, 50], [90, 255, 255]),
    "粉色": ([150, 85, 80], [175, 255, 255]),
    "浅橙色": ([0, 80, 90], [20, 162, 155]),
    "橙色": ([0, 165, 165], [30, 255, 255]),
}

#定义四种形状
shape_types = [
    "六边形",
    "梯形",
    "菱形",
    "三角形",
]

#轮廓线的颜色
draw_colors = {
    "绿色": (0, 255, 0),
    "粉色": (255, 0, 255),
    "浅橙色": (0, 165, 255),
    "橙色": (0, 128, 255),
}

#1. 读取图片
img = cv.imread("pictures/spare parts/1055.jpg")
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

#2. 转换为HSV图
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#3. 创建四个掩膜，识别四种颜色
mask_dict = {}
for name, (lower, upper) in color_types.items():
    lower_arr = np.array(lower)
    upper_arr = np.array(upper)
    mask = cv.inRange(hsv_img, lower_arr, upper_arr)
    mask_dict[name] = mask

    #进行物理降噪，填充内部孔洞，消去外部噪点
    mask_dict[name] = cv.morphologyEx(mask_dict[name], cv.MORPH_CLOSE, kernel)
    mask_dict[name] = cv.morphologyEx(mask_dict[name], cv.MORPH_OPEN, kernel)

result_img = img.copy()   # 在原图副本上绘制
label_count = 1            # 全局序号

for name, mask in mask_dict.items():
    # 查找外轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    color = draw_colors[name]   # 当前颜色对应的绘制颜色

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 500:          # 过滤太小的噪点（根据实际零件大小调整）
            continue

        # 多边形逼近，这里只是用于把轮廓画得更平滑，不改变顶点数
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        # 绘制轮廓
        cv.drawContours(result_img, [approx], -1, color, 3)

        # 计算轮廓的质心，作为标注位置
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 如果面积为零（极少情况），取边界框中心
            x, y, w, h = cv.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

        # 在质心附近标注序号
        cv.putText(result_img, str(label_count), (cx, cy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        label_count += 1

max_width = 1200
height, width = img.shape[:2]
scale = max_width / width if width > max_width else 1.0
new_width = int(width * scale)
new_height = int(height * scale)
display_img = cv.resize(result_img, (new_width, new_height))

cv.imshow("img", display_img)
cv.waitKey(0)
cv.destroyAllWindows()
