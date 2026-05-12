import cv2 as cv
import numpy as np


# （你的颜色阈值注释和类定义保持不变）
class Part:
    def __init__(self, color: str, shape: str):
        self.color = color
        self.shape = shape


color_types = {
    "绿色": ([35, 50, 50], [90, 255, 255]),
    "粉色": ([150, 85, 80], [175, 255, 255]),
    "浅橙色": ([0, 80, 90], [20, 162, 255]),
    "橙色": ([0, 165, 165], [30, 255, 255]),
}

draw_colors = {
    "绿色": (0, 255, 0),
    "粉色": (255, 0, 255),
    "浅橙色": (0, 165, 255),
    "橙色": (0, 128, 255),
}

target_vertices = {
    "橙色": 6,
    "浅橙色": 3,
    "绿色": 4,
    "粉色": 4,
}


# 修改了 fit_polygon 的逻辑：不再强制削减顶点数到 target_vert，
# 而是保留轮廓实际的近似形状，这样合并后的形状不会被破坏。
def fit_polygon(cnt, target_vert):
    """获取轮廓的凸包并进行近似，保留实际形状（不强制指定顶点数）"""
    hull = cv.convexHull(cnt)
    if hull is None:
        hull = cnt
    peri = cv.arcLength(hull, True)
    # 使用较小的 epsilon，保留更多细节
    epsilon = 0.02 * peri
    approx = cv.approxPolyDP(hull, epsilon, True)
    # 直接返回近似后的多边形，不强制压缩到 target_vert
    return approx.reshape(-1, 2)


# ---------- 主程序 ----------
img = cv.imread("pictures/spare parts/987.jpg")

# 【关键修改 1】把核改大！这样才能把挨着的色块“粘”在一起
kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

mask_dict = {}
for name, (lower, upper) in color_types.items():
    lower_arr = np.array(lower)
    upper_arr = np.array(upper)
    mask = cv.inRange(hsv_img, lower_arr, upper_arr)

    # 【关键修改 2】先用大核做闭运算（填缝），再做膨胀（扩大融合）
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)

    mask_dict[name] = mask

result_img = img.copy()
label_count = 1

for name, mask in mask_dict.items():
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    color = draw_colors[name]
    target_vert = target_vertices[name]

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 1000:
            continue

        # 获取融合后的大轮廓的近似多边形
        poly_pts = fit_polygon(cnt, target_vert)
        if poly_pts is None or len(poly_pts) < 3:
            continue

        # 绘制融合后的零件轮廓（闭合）
        poly_pts = poly_pts.reshape((-1, 1, 2))
        cv.drawContours(result_img, [poly_pts], -1, color, 3)

        # 计算整个零件（这个大轮廓）的质心
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

        # 只标记这一个零件标号，而不是标记每个小块
        cv.putText(result_img, str(label_count), (cx, cy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        label_count += 1

# 显示结果
max_width = 1200
height, width = img.shape[:2]
scale = max_width / width if width > max_width else 1.0
new_width = int(width * scale)
new_height = int(height * scale)
display_img = cv.resize(result_img, (new_width, new_height))

cv.imshow("img", display_img)
cv.waitKey(0)
cv.destroyAllWindows()