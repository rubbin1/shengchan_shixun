import cv2 as cv
import numpy as np

# 1. 读取图像
img = cv.imread("pictures/spare parts(without light)/2472.jpg")   # 替换为你的实际图片名
if img is None:
    print("图片未找到")
    exit()

# 2. 转换到 HSV 颜色空间
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 3. 你的最佳阈值（橙色/红色范围）
lower_orange = np.array([0, 170, 0])
upper_orange = np.array([20, 255, 255])

# 4. 创建掩膜并形态学处理
mask = cv.inRange(hsv, lower_orange, upper_orange)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # 填补内部孔洞
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # 去除外部噪点

# 5. 查找轮廓
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 6. 筛选六边形
for cnt in contours:
    area = cv.contourArea(cnt)
    if area < 2000:       # 根据目标大小调整，排除太小的干扰
        continue

    perimeter = cv.arcLength(cnt, True)
    # 多边形逼近，系数可以适当增大（0.04~0.06）以容忍边缘不光滑
    approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)

    # 六边形通常有6个顶点，但为了容错，可接受5~7个
    if 5 <= len(approx) <= 7:
        # 还可以添加额外条件：凸性、外接矩形长宽比等（如果需要）
        cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
        # 标记顶点
        for pt in approx:
            cv.circle(img, tuple(pt[0]), 5, (255, 0, 0), -1)

# 7. 等比缩放显示（避免变形）
max_width = 1200
height, width = img.shape[:2]
scale = max_width / width if width > max_width else 1.0
new_width = int(width * scale)
new_height = int(height * scale)
display_img = cv.resize(img, (new_width, new_height))

cv.imshow("Detected Orange Hexagon", display_img)
cv.waitKey(0)
cv.destroyAllWindows()