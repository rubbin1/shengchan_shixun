import cv2 as cv
import numpy as np

# ==================== 参数配置 ====================
IMAGE_PATH = "../pictures/spare parts/877.jpg"
MIN_AREA_RATIO = 0.0005
KERNEL_OPEN = (3, 3)
KERNEL_CLOSE = (3, 3)  # 一定不能大，否则会把相邻零件焊死
DIST_THRESH_RATIO = 0.3  # 距离变换阈值
SEPARATION_STRENGTH = 0.1  # 凹点分割强度(0~1)，越小越容易切
DEBUG_MODE = True
MAX_DISPLAY_WIDTH = 800

# ==================== 颜色阈值 ====================
color_types = {
    "绿色": ([35, 50, 50], [90, 255, 255]),
    "粉色": ([142, 111, 116], [175, 255, 255]),
    "浅橙色": ([0, 100, 145], [20, 167, 255]),
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


# ==================== 工具函数 ====================
def fit_polygon_adaptive(cnt, target_vert, max_trials=20):
    hull = cv.convexHull(cnt)
    if hull is None: hull = cnt
    peri = cv.arcLength(hull, True)
    best_approx, best_diff = None, 999
    for frac in np.linspace(0.01, 0.15, max_trials):
        approx = cv.approxPolyDP(hull, frac * peri, True)
        diff = abs(len(approx) - target_vert)
        if diff < best_diff:
            best_diff, best_approx = diff, approx
            if diff == 0: break
    return best_approx.reshape(-1, 2) if best_approx is not None else None


def split_contour_by_convexity_multi(cnt, strength=0.3):
    """
    增强版凹点分割：支持将一个大轮廓切成多个物体。
    strength: 凹点深度阈值系数，越小越敏感 (0.02~0.2 合适)
    """
    hull = cv.convexHull(cnt, returnPoints=False)
    if len(hull) < 3: return [cnt]
    defects = cv.convexityDefects(cnt, hull)
    if defects is None: return [cnt]

    x, y, w, h = cv.boundingRect(cnt)
    min_side = min(w, h)
    depth_thresh = strength * min_side  # 像素阈值

    deep_points = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > depth_thresh:
            deep_points.append(f)

    if len(deep_points) < 2:
        return [cnt]  # 凹点太少，不切

    # 获取凹点实际坐标
    pts = np.array([cnt[p][0] for p in deep_points])

    # 将凹点聚类：如果两个凹点距离很近，视为同一个切割位置（避免噪音）
    # 使用简单的距离聚类
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
    if len(clusters) < 2:  # 所有凹点挤在一起，无法形成有效切割线
        return [cnt]

    # 用每个聚类的中心点作为切割点
    cut_points = []
    for cluster in clusters:
        center = np.mean(pts[cluster], axis=0)
        cut_points.append(center)
    cut_points = np.array(cut_points)

    # 根据切割点把轮廓分成多个子轮廓
    # 方法：将轮廓点按离哪个切割点最近来划分区域，然后分别闭合
    cnt_pts = cnt.reshape(-1, 2)
    # 为每个切割点找轮廓上最近的点索引
    nearest_idxs = []
    for cp in cut_points:
        dists = np.linalg.norm(cnt_pts - cp, axis=1)
        nearest_idxs.append(np.argmin(dists))
    nearest_idxs = sorted(nearest_idxs)

    # 沿轮廓分段，每两个相邻切割点之间为一段，加上首尾闭合
    sub_contours = []
    for i in range(len(nearest_idxs)):
        start_idx = nearest_idxs[i]
        end_idx = nearest_idxs[(i + 1) % len(nearest_idxs)]
        if start_idx < end_idx:
            segment = cnt_pts[start_idx:end_idx + 1]
        else:
            segment = np.concatenate([cnt_pts[start_idx:], cnt_pts[:end_idx + 1]], axis=0)
        # 闭合
        if not np.array_equal(segment[0], segment[-1]):
            segment = np.vstack([segment, segment[0]])
        sub_contours.append(segment.reshape(-1, 1, 2).astype(np.int32))

    # 过滤面积太小的碎片
    final_contours = [c for c in sub_contours if cv.contourArea(c) > 100]
    return final_contours if final_contours else [cnt]


def advanced_separation(mask, dist_thresh_ratio=0.3, sep_strength=0.3, min_area=500):
    """
    组合策略：
    1. 先用距离变换分水岭尝试分离。
    2. 如果某个连通域分离后仍然只有一个物体，则用凹点分割再切一次。
    返回分离后的二值图。
    """
    # 距离变换分水岭
    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, dist_thresh_ratio * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(mask, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), markers)
    separated = np.zeros_like(mask)
    separated[markers > 1] = 255

    # 检查每个连通域是否真的被切开了
    num_labels, labels = cv.connectedComponents(mask)
    # 只处理面积较大的原始连通域
    for label_id in range(1, num_labels):
        orig_region = (labels == label_id).astype(np.uint8) * 255
        if cv.countNonZero(orig_region) < min_area:
            continue
        # 对应分离后的子区域数量
        sub_regions = cv.bitwise_and(separated, orig_region)
        sub_cnt, _ = cv.connectedComponents(sub_regions)
        # sub_cnt 包含背景(0)，所以子区域数 = sub_cnt - 1
        if sub_cnt - 1 <= 1:  # 没切开，还是1个物体
            # 对这个原始连通域提取轮廓，用凹点分割
            cnts, _ = cv.findContours(orig_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv.contourArea(cnt) < min_area: continue
                sub_cnts = split_contour_by_convexity_multi(cnt, strength=sep_strength)
                if len(sub_cnts) > 1:
                    # 从 separated 中抹掉原区域，再画上切割后的多个轮廓
                    separated = cv.bitwise_and(separated, cv.bitwise_not(orig_region))
                    cv.drawContours(separated, sub_cnts, -1, 255, -1)
    return separated


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
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * MIN_AREA_RATIO

    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_OPEN)
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_CLOSE)

    result_img = img.copy()
    label_count = 1

    for name, (lower, upper) in color_types.items():
        mask = cv.inRange(hsv_img, np.array(lower), np.array(upper))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)

        if DEBUG_MODE:
            cv.imshow(f"Mask - {name}", resize_to_display(mask, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        # 使用增强的分离策略（距离变换失败时自动凹点分割）
        separated = advanced_separation(mask, DIST_THRESH_RATIO, SEPARATION_STRENGTH, min_area)

        if DEBUG_MODE:
            cv.imshow(f"Separated - {name}", resize_to_display(separated, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        contours, _ = cv.findContours(separated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        color = draw_colors[name]
        target_vert = target_vertices.get(name, 4)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue
            poly_pts = fit_polygon_adaptive(cnt, target_vert)
            if poly_pts is None or len(poly_pts) < 3:
                hull = cv.convexHull(cnt)
                poly_pts = hull.reshape(-1, 2)
            poly_pts = poly_pts.reshape((-1, 1, 2))
            cv.drawContours(result_img, [poly_pts], -1, color, 3)

            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
            cv.putText(result_img, str(label_count), (cx, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            label_count += 1

    display = resize_to_display(result_img, MAX_DISPLAY_WIDTH)
    cv.imshow("Result", display)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()