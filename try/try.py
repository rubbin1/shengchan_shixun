import cv2 as cv
import numpy as np

# ==================== 可调参数 ====================
IMAGE_PATH = "../pictures/spare parts/1055.jpg"
MIN_AREA_ABS = 500
KERNEL_OPEN = (3, 3)
KERNEL_CLOSE = (1, 1)
DIST_THRESH_RATIO = 0.12      # 距离变换前景阈值
SEPARATION_STRENGTH = 0.03    # 凸缺陷深度系数（越小越敏感）
MIN_MARKER_AREA = 20          # 分水岭标记最小面积
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
shape_names_en = {3: "Triangle", 4: "Quadrangle", 6: "Hexagon"}


# ==================== 工具函数 ====================
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


def resize_to_display(img, max_width=800):
    h, w = img.shape[:2]
    if w <= max_width: return img
    return cv.resize(img, (max_width, int(h * (max_width / w))))


def smooth_contour(cnt, epsilon_factor=0.006):
    """平滑轮廓，减少噪声对凸缺陷检测的干扰"""
    peri = cv.arcLength(cnt, True)
    return cv.approxPolyDP(cnt, epsilon_factor * peri, True)


def split_contour_by_convexity_multi(cnt, strength=0.03):
    """用凸缺陷检测凹点，沿凹点切割粘连轮廓"""
    smoothed = smooth_contour(cnt)

    hull = cv.convexHull(smoothed, returnPoints=False)
    if len(hull) < 3:
        return [cnt]

    defects = cv.convexityDefects(smoothed, hull)
    if defects is None:
        return [cnt]

    x, y, w, h = cv.boundingRect(cnt)
    min_side = min(w, h)
    depth_thresh = strength * min_side

    # 收集所有足够深的凹点，记录 (轮廓索引, 深度)
    deep_points = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > depth_thresh:
            deep_points.append((f, depth))

    if len(deep_points) < 2:
        return [cnt]

    # 按深度降序排列，只保留最深的几个凹点（最多6个）
    deep_points.sort(key=lambda x: x[1], reverse=True)
    deep_points = deep_points[:4]
    pts = np.array([smoothed[p[0]][0] for p in deep_points])
    indices = [p[0] for p in deep_points]

    # 聚类：合并距离过近的凹点
    cluster_dist = min_side * 0.3
    clusters = []
    used = set()
    for i in range(len(pts)):
        if i in used:
            continue
        cl = [i]
        used.add(i)
        for j in range(i + 1, len(pts)):
            if j in used:
                continue
            if np.linalg.norm(pts[j] - pts[i]) < cluster_dist:
                cl.append(j)
                used.add(j)
        clusters.append(cl)

    if len(clusters) < 2:
        return [cnt]

    # 每个聚类取最深凹点的索引作为切割点（避免平均导致两个聚类合并到同一点）
    cut_indices = sorted(indices[cl[0]] for cl in clusters)
    # 去重：确保切割点之间的弧长超过轮廓周长的10%
    n_pts = len(smoothed)
    unique_cuts = [cut_indices[0]]
    for ci in cut_indices[1:]:
        dist_forward = (ci - unique_cuts[-1]) % n_pts
        dist_backward = (unique_cuts[-1] - ci) % n_pts
        if min(dist_forward, dist_backward) > n_pts * 0.1:
            unique_cuts.append(ci)
    cut_indices = unique_cuts

    # 沿切割点分割轮廓
    cnt_pts = smoothed.reshape(-1, 2)
    sub_contours = []
    for i in range(len(cut_indices)):
        start_idx = cut_indices[i]
        end_idx = cut_indices[(i + 1) % len(cut_indices)]
        if start_idx < end_idx:
            segment = cnt_pts[start_idx:end_idx + 1]
        else:
            segment = np.concatenate([cnt_pts[start_idx:], cnt_pts[:end_idx + 1]], axis=0)
        if len(segment) >= 3 and not np.array_equal(segment[0], segment[-1]):
            segment = np.vstack([segment, segment[0]])
        sub_contours.append(segment.reshape(-1, 1, 2).astype(np.int32))

    # 过滤无效切割：每个子轮廓的周长至少占原轮廓的15%
    total_peri = cv.arcLength(smoothed, True)
    valid_subs = []
    for sub in sub_contours:
        if cv.contourArea(sub) > 100 and cv.arcLength(sub, True) > total_peri * 0.15:
            valid_subs.append(sub)

    return valid_subs if len(valid_subs) >= 2 else [cnt]


def improved_watershed(mask, dist_thresh_ratio=0.12, min_marker_area=20):
    """改进的分水岭：用高阈值距离变换区域作为种子标记"""
    if np.count_nonzero(mask) < 50:
        return mask

    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)

    # 高阈值 → 只保留各零件的中心区域作为种子
    _, sure_fg = cv.threshold(dist, dist_thresh_ratio * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 轻度腐蚀以分离可能粘连的种子
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv.erode(sure_fg, kernel, iterations=1)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(sure_fg)

    # 过滤面积过小的种子
    clean_fg = np.zeros_like(sure_fg)
    valid_count = 0
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_marker_area:
            clean_fg[labels == i] = 255
            valid_count += 1

    if valid_count <= 1:
        return mask

    # 分水岭
    unknown = cv.subtract(mask, clean_fg)
    _, markers = cv.connectedComponents(clean_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.watershed(color_mask, markers)

    result = np.zeros_like(mask)
    result[markers > 1] = 255
    return result


def erosion_watershed_separation(mask):
    """
    自适应腐蚀+分水岭分离：
    逐步加深腐蚀来断开零件间的薄连接，找到分离效果最好的腐蚀级别。
    处理边-边贴合（凸缺陷无法检测的粘连）。
    返回分离后的轮廓列表。
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    best_contours = None
    best_count = 1

    for erode_iter in [1, 2, 3]:
        eroded = cv.erode(mask, kernel, iterations=erode_iter)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(eroded)

        valid = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(1, num_labels)
                 if stats[i, cv.CC_STAT_AREA] >= MIN_MARKER_AREA]

        if len(valid) <= 1:
            continue

        # 检测碎片化：最小组件面积 < 最大组件的15% → 腐蚀过度
        areas = [a for _, a in valid]
        if min(areas) < max(areas) * 0.15:
            break

        # 构建标记图像 —— 直接用腐蚀结果做种子（不膨胀，避免标记重新融合）
        sure_fg = np.zeros_like(mask, dtype=np.int32)
        for idx, (comp_id, _) in enumerate(valid):
            sure_fg[labels == comp_id] = idx + 1

        # 未知区域 = 原始mask - 标记区域
        unknown = cv.subtract(mask, np.uint8(sure_fg > 0) * 255)

        # 分水岭
        markers = sure_fg + 1  # 背景 = 1
        markers[unknown == 255] = 0
        cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), markers)

        # 逐标签提取轮廓（避免分水岭边界断裂导致区域融合）
        contours = []
        for label in range(2, markers.max() + 1):
            region = np.uint8(markers == label) * 255
            cnts, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv.contourArea(c) >= MIN_MARKER_AREA:
                    contours.append(c)

        if len(contours) > best_count:
            best_count = len(contours)
            best_contours = contours

    return best_contours


def separate_touching_parts(mask, min_area, separation_strength=0.03):
    """
    两级分离策略：
    1. 腐蚀+分水岭：断开薄连接（处理边-边贴合）
    2. 凸缺陷切割：仅对明确凹形(solidity<0.88)的剩余大轮廓
    """
    # --- 第一级：腐蚀+分水岭（主分离策略）---
    ws_contours = erosion_watershed_separation(mask)

    if ws_contours is not None:
        contours = ws_contours
    else:
        # 腐蚀分水岭无法分离，回退到原始mask找轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 估算典型单零件面积（用于第二级判断）
    areas = [cv.contourArea(c) for c in contours if cv.contourArea(c) >= min_area]
    median_area = np.median(areas) if len(areas) >= 3 else min_area * 2
    merge_threshold = max(median_area * 1.5, min_area * 3)

    all_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        # 面积正常的直接保留
        if area <= merge_threshold:
            all_contours.append(cnt)
            continue

        # --- 第二级：凸缺陷切割（仅用于有粘连证据的大轮廓）---
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0

        # 统计深凹点用于区分真实粘连 vs 轮廓噪声
        x, y, w, h = cv.boundingRect(cnt)
        min_side = min(w, h)
        smoothed_chk = smooth_contour(cnt)
        hull_idx_chk = cv.convexHull(smoothed_chk, returnPoints=False)
        n_deep = 0
        if len(hull_idx_chk) >= 3:
            defects_chk = cv.convexityDefects(smoothed_chk, hull_idx_chk)
            if defects_chk is not None:
                for j in range(defects_chk.shape[0]):
                    _, _, _, d = defects_chk[j, 0]
                    if d / 256.0 > separation_strength * min_side:
                        n_deep += 1

        # 分离条件：
        # - solidity > 0.95: 几乎完美凸形 → 不拆分（防止单零件误拆）
        # - solidity < 0.88: 明确凹形 → 拆分
        # - solidity 0.88~0.95 + 2~4深凹点: 边-边贴合 → 拆分
        # - solidity > 0.92 + 5个以上凹点: 轮廓噪声 → 不拆分
        if solidity > 0.95:
            should_split = False
        elif solidity < 0.88:
            should_split = True
        elif 2 <= n_deep <= 4:
            should_split = True
        else:
            should_split = False
        if solidity > 0.92 and n_deep > 5:
            should_split = False

        if should_split:
            sub_cnts = split_contour_by_convexity_multi(cnt, separation_strength)
            if len(sub_cnts) > 1:
                sub_areas = [cv.contourArea(s) for s in sub_cnts if cv.contourArea(s) >= min_area]
                if len(sub_areas) >= 2:
                    ratio = min(sub_areas) / max(sub_areas)
                    if ratio < 0.2:
                        all_contours.append(cnt)
                        continue
                for sub in sub_cnts:
                    if cv.contourArea(sub) >= min_area:
                        all_contours.append(sub)
                continue

        all_contours.append(cnt)

    return all_contours


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
        # 1. 颜色掩膜提取 + 形态学清理
        mask = cv.inRange(hsv_img, np.array(lower), np.array(upper))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

        if DEBUG_MODE:
            cv.imshow(f"Mask - {name}", resize_to_display(mask, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        # 2. 多策略分离粘连零件
        contours = separate_touching_parts(mask, min_area, SEPARATION_STRENGTH)

        if DEBUG_MODE:
            # 画出分离后的轮廓供调试
            sep_viz = np.zeros_like(mask)
            cv.drawContours(sep_viz, contours, -1, 255, -1)
            cv.imshow(f"Separated - {name}", resize_to_display(sep_viz, MAX_DISPLAY_WIDTH))
            cv.waitKey(1)

        # 3. 当前颜色对应的标签信息
        en_color = color_name_en.get(name, name)
        target_vert = target_vertices.get(name, 4)
        en_shape = shape_names_en.get(target_vert, f"{target_vert}gon")

        # 4. 遍历每个零件并标注
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            cv.circle(result_img, (cx, cy), 3, (0, 0, 0), -1)
            text = f"{en_color} {en_shape}"
            cv.putText(result_img, text, (cx + 6, cy - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.imshow("Result", resize_to_display(result_img, MAX_DISPLAY_WIDTH))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
