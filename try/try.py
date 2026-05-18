import cv2 as cv
import numpy as np

# ==================== 可调参数 ====================
IMAGE_PATH = "../pictures/spare parts/927.jpg"
MIN_AREA_ABS = 500          # 绝对最小面积（针对800x600图像）
# 修复：KERNEL_CLOSE 必须大于 1，否则无法闭合孔洞
KERNEL_OPEN = (3, 3)
KERNEL_CLOSE = (3, 3)
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
def fit_polygon_exact(cnt, target_vert):
    """
    强制拟合多边形到精确的顶点数 (target_vert)
    使用二分查找法找到最合适的 epsilon 值
    """
    hull = cv.convexHull(cnt)
    if hull is None:
        hull = cnt

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

    if len(best_approx) > target_vert:
        pts = best_approx.reshape(-1, 2)
        dists = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dists.append(np.linalg.norm(p1 - p2))

        sorted_idx = np.argsort(dists)[::-1]
        selected_indices = []
        for idx in sorted_idx:
            if len(selected_indices) >= target_vert:
                break
            if all(abs(idx - s) % len(pts) != 1 for s in selected_indices):
                selected_indices.append(idx)

        selected_indices.sort()
        return pts[selected_indices]
    elif len(best_approx) < target_vert:
        return hull.reshape(-1, 2)
    else:
        return best_approx.reshape(-1, 2)

def refine_quad_contour(cnt, target_vert=4):
    """
    强制拟合四边形，保留菱形/梯形的角度，而不是画成外接矩形。
    """
    # 如果不是4边形，交给通用函数处理（比如三角形、六边形）
    if target_vert != 4:
        return fit_polygon_exact(cnt, target_vert)

    # 通用四边拟合法：让 approxPolyDP 强制寻找最接近的4个点
    hull = cv.convexHull(cnt)
    peri = cv.arcLength(hull, True)

    # 微调 epsilon 范围，找到最接近4边形状的结果
    best_approx = None
    for frac in np.linspace(0.02, 0.12, 15):
        approx = cv.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2)
        elif len(approx) == 5 and best_approx is None:
            best_approx = approx
        elif len(approx) == 6 and best_approx is None:
            best_approx = approx

    # 如果没找到完美4边形，尝试5边形或6边形中找最长4条边进行压缩
    if best_approx is not None:
        pts = best_approx.reshape(-1, 2)
        dists = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i+1) % len(pts)]
            dists.append(np.linalg.norm(p1 - p2))

        # 找最长的4条边对应的顶点
        sorted_idx = np.argsort(dists)[::-1]
        selected = []
        for idx in sorted_idx:
            if len(selected) >= 4: break
            # 防止选到相邻点
            if all(abs(idx - s) % len(pts) != 1 for s in selected):
                selected.append(idx)
        selected.sort()
        return pts[selected]

    # 如果都失败，退回凸包
    return hull.reshape(-1, 2)

# ... 原有的 split_contour_by_convexity_multi, enhanced_watershed 等函数保持原样 ...
def split_contour_by_convexity_multi(cnt, strength=0.3):
    """基于凸缺陷的凹点分割（可处理多个物体）"""
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

    if len(deep_points) < 2:
        return [cnt]

    pts = np.array([cnt[p][0] for p in deep_points])

    def cluster_points(pts, dist_thresh=20):
        clusters = []
        used = set()
        for i, p in enumerate(pts):
            if i in used: continue
            cluster = [i]
            used.add(i)
            for j in range(i+1, len(pts)):
                if j in used: continue
                if np.linalg.norm(pts[j]-p) < dist_thresh:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        return clusters

    clusters = cluster_points(pts, dist_thresh=min_side*0.5)
    if len(clusters) < 2:
        return [cnt]

    cut_points = np.array([np.mean(pts[cl], axis=0) for cl in clusters])
    cnt_pts = cnt.reshape(-1, 2)
    nearest_idxs = sorted([np.argmin(np.linalg.norm(cnt_pts-cp, axis=1)) for cp in cut_points])

    sub_contours = []
    for i in range(len(nearest_idxs)):
        start_idx = nearest_idxs[i]
        end_idx = nearest_idxs[(i+1) % len(nearest_idxs)]
        if start_idx < end_idx:
            segment = cnt_pts[start_idx:end_idx+1]
        else:
            segment = np.concatenate([cnt_pts[start_idx:], cnt_pts[:end_idx+1]], axis=0)
        if not np.array_equal(segment[0], segment[-1]):
            segment = np.vstack([segment, segment[0]])
        sub_contours.append(segment.reshape(-1, 1, 2).astype(np.int32))

    return [c for c in sub_contours if cv.contourArea(c) > 100] or [cnt]

def enhanced_watershed(mask, dist_thresh_ratio=0.15, min_marker_area=20):
    """迭代腐蚀前景标记，强制分离后用于分水岭"""
    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, dist_thresh_ratio * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    prev_count = 1
    while True:
        eroded = cv.erode(sure_fg, kernel, iterations=1)
        num_labels, _ = cv.connectedComponents(eroded)
        if num_labels <= prev_count or num_labels <= 2:
            break
        sure_fg = eroded
        prev_count = num_labels
        if num_labels > 20:
            break

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

def zhang_suen_skeleton(mask):
    """纯 OpenCV 实现的 Zhang-Suen 骨架化算法"""
    img = mask.copy() // 255
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skeleton = cv.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv.countNonZero(img) == 0:
            break
    return skeleton * 255

def skeleton_cut(mask):
    """利用骨架交叉点切割粘连区域"""
    skeleton = zhang_suen_skeleton(mask)
    if cv.countNonZero(skeleton) < 10:
        return mask

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbors = cv.filter2D(skeleton, cv.CV_8U, kernel)
    cross_points = (neighbors >= 13) & (skeleton > 0)
    cross_idx = np.argwhere(cross_points)

    if len(cross_idx) < 2:
        return mask

    pts = cross_idx[:, ::-1]
    max_dist = 0
    pair = (0, 1)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                pair = (i, j)
    p1, p2 = tuple(pts[pair[0]]), tuple(pts[pair[1]])

    cut_mask = mask.copy()
    cv.line(cut_mask, p1, p2, color=0, thickness=3)
    return cut_mask

def force_separation(mask, kernel_size=3, iterations=2):
    """
    通过迭代腐蚀来物理断开粘连区域。
    kernel_size: 腐蚀核大小，建议3-7
    iterations: 腐蚀次数，根据图片分辨率调整
    """
    # 1. 第一次膨胀，让轮廓更加清晰，以便后续腐蚀断开更精准
    # 防止过度腐蚀导致丢失形状
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv.erode(mask, kernel, iterations=iterations)
    # 2. 利用距离变换找到每个连通区域的“核心”
    dist = cv.distanceTransform(eroded, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, 0.2 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    final_mask = cv.dilate(sure_fg, kernel, iterations=iterations)
    return final_mask

def resize_to_display(img, max_width=800):
    h, w = img.shape[:2]
    if w <= max_width: return img
    return cv.resize(img, (max_width, int(h * (max_width / w))))


def restore_shapes_fixed(original_mask, separated_mask):
    """
    1. 先找出 separated_mask 中的所有独立零件（连通分量）。
    2. 为每个独立零件单独从原始掩膜中恢复形状。
    """
    # 1. 获取 separated_mask 中所有独立的连通分量
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(separated_mask, connectivity=8)

    final_mask = np.zeros_like(original_mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    for i in range(1, num_labels):  # 跳过背景(0)
        # 2. 创建当前零件的独立掩膜
        comp_mask = (labels == i).astype(np.uint8) * 255

        # 3. 计算当前零件的边界框
        x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_WIDTH], stats[
            i, cv.CC_STAT_HEIGHT]

        # 4. 裁剪出对应的ROI（局部区域）
        roi_original = original_mask[y:y + h, x:x + w]
        roi_separated = comp_mask[y:y + h, x:x + w]

        # 5. 关键修改：只膨胀1次，找回腐蚀掉的真实边缘（减少到1次，防止粘连）
        roi_separated = cv.dilate(roi_separated, kernel, iterations=1)

        # 6. 与原掩膜取交集，完美恢复形状
        roi_restored = cv.bitwise_and(roi_original, roi_separated)

        # 7. 将恢复好的零件放回最终掩膜
        final_mask[y:y + h, x:x + w] = cv.bitwise_or(final_mask[y:y + h, x:x + w], roi_restored)
    return final_mask

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
    label_count = 1

    for name, (lower, upper) in color_types.items():
        mask = cv.inRange(hsv_img, np.array(lower), np.array(upper))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

        # ================== 核心修改：加大腐蚀力度 ==================
        # 针对浅橙色（三角形），单独加大腐蚀力度，彻底断开粘连线
        if name == "浅橙色":
            # 使用更大的核(5x5)和更多迭代(3次)来强制断开粘连
            force_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            # 先做一次强力腐蚀（断开粘连），再做一次距离变换恢复核心
            eroded = cv.erode(mask, force_kernel, iterations=3)
            dist = cv.distanceTransform(eroded, cv.DIST_L2, 5)
            _, sure_fg = cv.threshold(dist, 0.15 * dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            separated = cv.dilate(sure_fg, force_kernel, iterations=2)
        else:
            # 其他颜色保持原来的分离策略
            separated = force_separation(mask, kernel_size=3, iterations=2)
        # ==========================================================

        # 核心：恢复形状（用我们优化过的独立恢复版本）
        separated = restore_shapes_fixed(mask, separated)

        # 分水岭细化（可选，让粘更紧密的部分彻底断开）
        separated = enhanced_watershed(separated, dist_thresh_ratio=0.05, min_marker_area=20)

        if DEBUG_MODE:
            cv.imshow(f"Separated - {name}", resize_to_display(separated, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        contours, _ = cv.findContours(separated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area: continue

            color = draw_colors[name]
            target_vert = target_vertices.get(name, 4)

            x, y, w, h = cv.boundingRect(cnt)

            # 精确拟合（你的refine_quad_contour已经支持三角形，会自动适配）
            poly_pts = refine_quad_contour(cnt, target_vert)

            if poly_pts is None or len(poly_pts) < 3:
                hull = cv.convexHull(cnt)
                poly_pts = hull.reshape(-1, 2)
            poly_pts = poly_pts.reshape((-1, 1, 2))

            cv.drawContours(result_img, [poly_pts], -1, color, 2)

            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            cv.putText(result_img, str(label_count), (cx, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            label_count += 1

    cv.imshow("Result", resize_to_display(result_img, MAX_DISPLAY_WIDTH))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()