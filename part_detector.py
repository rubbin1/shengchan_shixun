# part_detector.py
import cv2 as cv
import numpy as np


class PartDetector:
    def __init__(self, min_area_abs=500, min_area_ratio=None, debug=False):
        # 颜色阈值
        self.color_types = {
            "绿色": ([35, 50, 50], [90, 255, 255]),
            "粉色": ([142, 111, 116], [175, 255, 255]),
            "浅橙色": ([0, 100, 145], [20, 167, 255]),
            "橙色": ([0, 165, 165], [30, 255, 255]),
        }
        self.draw_colors = {
            "绿色": (0, 255, 0),
            "粉色": (255, 0, 255),
            "浅橙色": (0, 165, 255),
            "橙色": (0, 128, 255),
        }
        self.color_name_en = {
            "绿色": "Green",
            "粉色": "Pink",
            "浅橙色": "LtOrange",
            "橙色": "Orange",
        }
        self.target_vertices = {
            "橙色": 6,
            "浅橙色": 3,
            "绿色": 4,
            "粉色": 4,
        }
        self.shape_names_en = {3: "Triangle", 4: "Quadrangle", 6: "Hexagon"}

        self.min_area_abs = min_area_abs
        self.min_area_ratio = min_area_ratio
        self.debug = debug

        # 形态学核
        self.kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))

        # 分离参数
        self.sep_strength = 0.03       # 凸缺陷深度系数（越小越敏感）
        self.min_marker_area = 20      # 分水岭标记最小面积

    # ==================== 主检测接口 ====================
    def detect(self, bgr_image):
        """输入 BGR 图像，返回标注了颜色+形状标签的 BGR 图像"""
        img = bgr_image.copy()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]

        # 面积阈值：优先使用 ratio，否则用绝对值按图像尺寸缩放
        if self.min_area_ratio is not None:
            min_area = self.min_area_ratio * img_w * img_h
        else:
            min_area = self.min_area_abs * (img_w * img_h) / (800 * 600)

        result_img = img.copy()

        for name, (lower, upper) in self.color_types.items():
            # 1. 颜色掩膜 + 形态学清理
            mask = cv.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel_close)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel_open)

            # 2. 两级分离：腐蚀分水岭（主） + 凸缺陷切割（回退）
            contours = self._separate_touching_parts(mask, min_area)

            # 3. 标签信息
            en_color = self.color_name_en.get(name, name)
            target_vert = self.target_vertices.get(name, 4)
            en_shape = self.shape_names_en.get(target_vert, f"{target_vert}gon")

            # 4. 遍历每个零件画质心和标签
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

        return result_img

    # ==================== 两级分离策略 ====================
    def _separate_touching_parts(self, mask, min_area):
        """
        两级分离：
        1. 腐蚀+分水岭：断开薄连接（处理边-边贴合）
        2. 凸缺陷切割：仅对明确凹形(solidity<0.88)或2~4个深凹点的大轮廓
        """
        # --- 第一级：腐蚀+分水岭 ---
        ws_contours = self._erosion_watershed_separation(mask)

        if ws_contours is not None:
            contours = ws_contours
        else:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        areas = [cv.contourArea(c) for c in contours if cv.contourArea(c) >= min_area]
        median_area = np.median(areas) if len(areas) >= 3 else min_area * 2
        merge_threshold = max(median_area * 1.5, min_area * 3)

        all_contours = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue

            if area <= merge_threshold:
                all_contours.append(cnt)
                continue

            # --- 第二级：凸缺陷切割 ---
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 1.0

            x, y, w, h = cv.boundingRect(cnt)
            min_side = min(w, h)
            smoothed_chk = self._smooth_contour(cnt)
            hull_idx_chk = cv.convexHull(smoothed_chk, returnPoints=False)
            n_deep = 0
            if len(hull_idx_chk) >= 3:
                defects_chk = cv.convexityDefects(smoothed_chk, hull_idx_chk)
                if defects_chk is not None:
                    for j in range(defects_chk.shape[0]):
                        _, _, _, d = defects_chk[j, 0]
                        if d / 256.0 > self.sep_strength * min_side:
                            n_deep += 1

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
                sub_cnts = self._split_contour_by_convexity_multi(cnt)
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

    # ==================== 腐蚀+分水岭分离 ====================
    def _erosion_watershed_separation(self, mask):
        """
        自适应腐蚀+分水岭：逐步加深腐蚀断开薄连接，
        找到分离效果最好的级别。返回轮廓列表或 None。
        """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        best_contours = None
        best_count = 1

        for erode_iter in [1, 2, 3]:
            eroded = cv.erode(mask, kernel, iterations=erode_iter)
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(eroded)

            valid = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(1, num_labels)
                     if stats[i, cv.CC_STAT_AREA] >= self.min_marker_area]

            if len(valid) <= 1:
                continue

            areas = [a for _, a in valid]
            if min(areas) < max(areas) * 0.15:
                break

            # 构建标记（不膨胀，避免标记重新融合）
            sure_fg = np.zeros_like(mask, dtype=np.int32)
            for idx, (comp_id, _) in enumerate(valid):
                sure_fg[labels == comp_id] = idx + 1

            unknown = cv.subtract(mask, np.uint8(sure_fg > 0) * 255)

            markers = sure_fg + 1
            markers[unknown == 255] = 0
            cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), markers)

            # 逐标签提取轮廓（避免分水岭边界断裂导致融合）
            contours = []
            for label in range(2, markers.max() + 1):
                region = np.uint8(markers == label) * 255
                cnts, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv.contourArea(c) >= self.min_marker_area:
                        contours.append(c)

            if len(contours) > best_count:
                best_count = len(contours)
                best_contours = contours

        return best_contours

    # ==================== 凸缺陷切割 ====================
    def _smooth_contour(self, cnt, epsilon_factor=0.006):
        peri = cv.arcLength(cnt, True)
        return cv.approxPolyDP(cnt, epsilon_factor * peri, True)

    def _split_contour_by_convexity_multi(self, cnt):
        """用凸缺陷检测凹点，沿凹点切割粘连轮廓"""
        smoothed = self._smooth_contour(cnt)

        hull = cv.convexHull(smoothed, returnPoints=False)
        if len(hull) < 3:
            return [cnt]

        defects = cv.convexityDefects(smoothed, hull)
        if defects is None:
            return [cnt]

        x, y, w, h = cv.boundingRect(cnt)
        min_side = min(w, h)
        depth_thresh = self.sep_strength * min_side

        deep_points = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            if depth > depth_thresh:
                deep_points.append((f, depth))

        if len(deep_points) < 2:
            return [cnt]

        deep_points.sort(key=lambda x: x[1], reverse=True)
        deep_points = deep_points[:4]
        pts = np.array([smoothed[p[0]][0] for p in deep_points])
        indices = [p[0] for p in deep_points]

        # 聚类
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

        # 每个聚类取最深凹点索引
        cut_indices = sorted(indices[cl[0]] for cl in clusters)
        # 去重
        n_pts = len(smoothed)
        unique_cuts = [cut_indices[0]]
        for ci in cut_indices[1:]:
            dist_forward = (ci - unique_cuts[-1]) % n_pts
            dist_backward = (unique_cuts[-1] - ci) % n_pts
            if min(dist_forward, dist_backward) > n_pts * 0.1:
                unique_cuts.append(ci)
        cut_indices = unique_cuts

        # 分割轮廓
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

        # 过滤无效切割
        total_peri = cv.arcLength(smoothed, True)
        valid_subs = []
        for sub in sub_contours:
            if cv.contourArea(sub) > 100 and cv.arcLength(sub, True) > total_peri * 0.15:
                valid_subs.append(sub)

        return valid_subs if len(valid_subs) >= 2 else [cnt]
