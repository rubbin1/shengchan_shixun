# part_detector.py
import cv2 as cv
import numpy as np

class PartDetector:
    def __init__(self, min_area_ratio=0.0005, debug=False):
        # 颜色阈值（可根据实际相机图像微调）
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
        self.target_vertices = {
            "橙色": 6,
            "浅橙色": 3,
            "绿色": 4,
            "粉色": 4,
        }
        self.min_area_ratio = min_area_ratio
        self.debug = debug

        # 形态学核
        self.kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.dist_thresh_ratio = 0.3
        self.sep_strength = 0.1

    def detect(self, bgr_image):
        """输入 BGR 图像，返回绘制了轮廓和编号的 BGR 图像"""
        img = bgr_image.copy()
        h, w = img.shape[:2]
        img_area = h * w
        min_area = img_area * self.min_area_ratio

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        result_img = img.copy()
        label_count = 1

        for name, (lower, upper) in self.color_types.items():
            mask = cv.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel_open)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel_close)

            # 使用增强分离（距离变换+凹点分割）
            separated = self._advanced_separation(mask, min_area)

            contours, _ = cv.findContours(separated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            color = self.draw_colors[name]
            target_vert = self.target_vertices.get(name, 4)

            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < min_area:
                    continue

                # 多边形拟合
                poly_pts = self._fit_polygon(cnt, target_vert)
                if poly_pts is None or len(poly_pts) < 3:
                    hull = cv.convexHull(cnt)
                    poly_pts = hull.reshape(-1, 2)
                poly_pts = poly_pts.reshape((-1, 1, 2))
                cv.drawContours(result_img, [poly_pts], -1, color, 3)

                # 质心标号
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                else:
                    x, y, wb, hb = cv.boundingRect(cnt)
                    cx, cy = x + wb // 2, y + hb // 2
                cv.putText(result_img, str(label_count), (cx, cy),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                label_count += 1

        return result_img

    # ---------- 以下为辅助函数（直接从原代码移植）----------
    def _fit_polygon(self, cnt, target_vert, max_trials=20):
        hull = cv.convexHull(cnt)
        if hull is None:
            hull = cnt
        peri = cv.arcLength(hull, True)
        best_approx, best_diff = None, 999
        for frac in np.linspace(0.01, 0.15, max_trials):
            approx = cv.approxPolyDP(hull, frac * peri, True)
            diff = abs(len(approx) - target_vert)
            if diff < best_diff:
                best_diff, best_approx = diff, approx
                if diff == 0:
                    break
        return best_approx.reshape(-1, 2) if best_approx is not None else None

    def _advanced_separation(self, mask, min_area):
        # 距离变换+分水岭
        dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist, self.dist_thresh_ratio * dist.max(), 255, cv.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(mask, sure_fg)
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), markers)
        separated = np.zeros_like(mask)
        separated[markers > 1] = 255

        # 凹点二次分割（对仍粘连的区域）
        num_labels, labels = cv.connectedComponents(mask)
        for label_id in range(1, num_labels):
            orig_region = (labels == label_id).astype(np.uint8) * 255
            if cv.countNonZero(orig_region) < min_area:
                continue
            sub_regions = cv.bitwise_and(separated, orig_region)
            sub_cnt, _ = cv.connectedComponents(sub_regions)
            if sub_cnt - 1 <= 1:  # 没切开
                cnts, _ = cv.findContours(orig_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv.contourArea(cnt) < min_area:
                        continue
                    sub_cnts = self._split_contour_by_convexity(cnt)
                    if len(sub_cnts) > 1:
                        separated = cv.bitwise_and(separated, cv.bitwise_not(orig_region))
                        cv.drawContours(separated, sub_cnts, -1, 255, -1)
        return separated

    def _split_contour_by_convexity(self, cnt):
        strength = self.sep_strength
        hull = cv.convexHull(cnt, returnPoints=False)
        if len(hull) < 3:
            return [cnt]
        defects = cv.convexityDefects(cnt, hull)
        if defects is None:
            return [cnt]
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
        # 简单的空间聚类
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