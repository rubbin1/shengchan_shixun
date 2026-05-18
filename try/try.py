import cv2 as cv
import numpy as np

# ==================== 可调参数 ====================
IMAGE_PATH = "../pictures/spare parts/485.jpg"
MIN_AREA_ABS = 500          # 绝对最小面积（针对800x600图像）
KERNEL_OPEN = (3, 3)
KERNEL_CLOSE = (1, 1)       # 尽可能小，避免将零件焊死
DIST_THRESH_RATIO = 0.1     # 距离变换前景阈值（越小越容易分割）
SEPARATION_STRENGTH = 0.02  # 凹点深度系数（越小越敏感）
MIN_MARKER_AREA = 20        # 分水岭标记最小面积
DEBUG_MODE = True
MAX_DISPLAY_WIDTH = 800

# ==================== 颜色阈值 ====================
color_types = {
    "绿色": ([35, 50, 50], [90, 255, 255]),
    "粉色": ([142, 111, 116], [175, 255, 255]),
    "浅橙色": ([0, 100, 145], [20, 167, 255]),
    "橙色": ([0, 165, 165], [30, 255, 255]),
}
# 绘图颜色（BGR）
draw_colors = {
    "绿色": (0, 255, 0),
    "粉色": (255, 0, 255),
    "浅橙色": (0, 165, 255),
    "橙色": (0, 128, 255),
}
# 英文名称映射
color_name_en = {
    "绿色": "Green",
    "粉色": "Pink",
    "浅橙色": "LtOrange",
    "橙色": "Orange",
}
target_vertices = {
    "橙色": 6,
    "浅橙色": 3,
    "绿色": 4,
    "粉色": 4,
}
# 英文形状名称
shape_names_en = {3: "Tri", 4: "Quad", 5: "Penta", 6: "Hexa", 7: "Hepta", 8: "Octa"}

# ==================== 工具函数（保持不变） ====================
def fit_polygon_exact(cnt, target_vert):
    hull = cv.convexHull(cnt)
    if hull is None: hull = cnt
    peri = cv.arcLength(hull, True)
    if len(hull) <= target_vert:
        return hull.reshape(-1, 2)
    low, high = 0.001, 0.5
    best_approx = None
    for _ in range(30):
        mid = (low + high) / 2
        epsilon = mid * peri
        approx = cv.approxPolyDP(hull, epsilon, True)
        if len(approx) == target_vert:
            return approx.reshape(-1, 2)
        elif len(approx) > target_vert:
            low = mid
        else:
            high = mid
        best_approx = approx
    if best_approx is not None and len(best_approx) >= 3:
        return best_approx.reshape(-1, 2)
    return hull.reshape(-1, 2)


def split_contour_by_convexity_multi(cnt, strength=0.3):
    hull = cv.convexHull(cnt, returnPoints=False)
    if len(hull) < 3: return [cnt]
    defects = cv.convexityDefects(cnt, hull)
    if defects is None: return [cnt]
    x, y, w, h = cv.boundingRect(cnt)
    min_side = min(w, h)
    depth_thresh = strength * min_side
    deep_points = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > depth_thresh:
            deep_points.append(f)
    if len(deep_points) < 2: return [cnt]
    pts = np.array([cnt[p][0] for p in deep_points])

    def cluster_points(pts, dist_thresh=20):
        clusters = []
        used = set()
        for i, p in enumerate(pts):
            if i in used: continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(pts)):
                if j in used: continue
                if np.linalg.norm(pts[j] - p) < dist_thresh:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        return clusters

    clusters = cluster_points(pts, dist_thresh=min_side * 0.5)
    if len(clusters) < 2: return [cnt]
    cut_points = np.array([np.mean(pts[cl], axis=0) for cl in clusters])
    cnt_pts = cnt.reshape(-1, 2)
    nearest_idxs = sorted([np.argmin(np.linalg.norm(cnt_pts - cp, axis=1)) for cp in cut_points])
    sub_contours = []
    for i in range(len(nearest_idxs)):
        start_idx = nearest_idxs[i]
        end_idx = nearest_idxs[(i + 1) % len(nearest_idxs)]
        if start_idx < end_idx:
            segment = cnt_pts[start_idx:end_idx + 1]
        else:
            segment = np.concatenate([cnt_pts[start_idx:], cnt_pts[:end_idx + 1]], axis=0)
        if not np.array_equal(segment[0], segment[-1]):
            segment = np.vstack([segment, segment[0]])
        sub_contours.append(segment.reshape(-1, 1, 2).astype(np.int32))
    return [c for c in sub_contours if cv.contourArea(c) > 100] or [cnt]


def enhanced_watershed(mask, dist_thresh_ratio=0.15, min_marker_area=20):
    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, dist_thresh_ratio * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    prev_count = 1
    while True:
        eroded = cv.erode(sure_fg, kernel, iterations=1)
        num_labels, _ = cv.connectedComponents(eroded)
        if num_labels <= prev_count or num_labels <= 2: break
        sure_fg = eroded
        prev_count = num_labels
        if num_labels > 20: break
    sure_fg = cv.dilate(sure_fg, kernel, iterations=1)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(sure_fg)
    clean_fg = np.zeros_like(sure_fg)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_marker_area:
            clean_fg[labels == i] = 255
    unknown = cv.subtract(mask, clean_fg)
    _, markers = cv.connectedComponents(clean_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), markers)
    separated = np.zeros_like(mask)
    separated[markers > 1] = 255
    return separated


def force_separation(mask, kernel_size=3, iterations=2):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv.erode(mask, kernel, iterations=iterations)
    dist = cv.distanceTransform(eroded, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, 0.2 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    final_mask = cv.dilate(sure_fg, kernel, iterations=iterations)
    return final_mask


def restore_shapes(original_mask, separated_mask):
    contours, _ = cv.findContours(separated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(original_mask)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        crop_original = original_mask[y:y + h, x:x + w]
        crop_separated = separated_mask[y:y + h, x:x + w]
        kernel = np.ones((3, 3), np.uint8)
        crop_separated = cv.dilate(crop_separated, kernel, iterations=2)
        restored_crop = cv.bitwise_and(crop_original, crop_separated)
        final_mask[y:y + h, x:x + w] = cv.bitwise_or(final_mask[y:y + h, x:x + w], restored_crop)
    return final_mask


def resize_to_display(img, max_width=800):
    h, w = img.shape[:2]
    if w <= max_width: return img
    return cv.resize(img, (max_width, int(h * (max_width / w))))


# ==================== 主程序 ====================
def main():
    img = cv.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: 无法读取 {IMAGE_PATH}")
        return
    img = resize_to_display(img, MAX_DISPLAY_WIDTH)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]
    min_area = MIN_AREA_ABS * (img_w * img_h) / (800 * 600)

    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_OPEN)
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_CLOSE)

    result_img = img.copy()

    for name, (lower, upper) in color_types.items():
        # 1. 颜色掩膜提取
        mask = cv.inRange(hsv_img, np.array(lower), np.array(upper))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

        # 2. 强制分离（轻度腐蚀断开粘连）
        separated = force_separation(mask, kernel_size=3, iterations=2)
        # 3. 捞回原始形状
        separated = restore_shapes(mask, separated)
        # 4. 分水岭细化
        separated = enhanced_watershed(separated, dist_thresh_ratio=0.05, min_marker_area=20)

        if DEBUG_MODE:
            cv.imshow(f"Separated - {name}", resize_to_display(separated, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        # 5. 提取轮廓
        contours, _ = cv.findContours(separated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 当前颜色对应的英文名和形状名
        en_color = color_name_en.get(name, name)
        target_vert = target_vertices.get(name, 4)
        en_shape = shape_names_en.get(target_vert, f"{target_vert}gon")

        # 6. 遍历每个零件
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue

            # 计算质心
            x, y, w, h = cv.boundingRect(cnt)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # 画质心小圆点（黑色）
            cv.circle(result_img, (cx, cy), 3, (0, 0, 0), -1)

            # 英文标签（黑色，字号 0.5，粗细 1）
            text = f"{en_color} {en_shape}"
            cv.putText(result_img, text, (cx + 6, cy - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.imshow("Result", resize_to_display(result_img, MAX_DISPLAY_WIDTH))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()