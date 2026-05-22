# part_detector.py
import cv2 as cv
import numpy as np

class PartDetector:
    def __init__(self, min_area_abs=500, debug=False):
        # 颜色阈值（可根据实际相机图像微调）
        self.color_types = {
            "绿色": ([35, 50, 50], [90, 255, 255]),
            "粉色": ([142, 111, 116], [175, 255, 255]),
            "浅橙色": ([0, 100, 145], [20, 167, 255]),
            "橙色": ([0, 165, 165], [30, 255, 255]),
        }
        # 绘图颜色（BGR，仅用于调试，实际标注用黑色）
        self.draw_colors = {
            "绿色": (0, 255, 0),
            "粉色": (255, 0, 255),
            "浅橙色": (0, 165, 255),
            "橙色": (0, 128, 255),
        }
        # 英文颜色名
        self.color_name_en = {
            "绿色": "Green",
            "粉色": "Pink",
            "浅橙色": "LtOrange",
            "橙色": "Orange",
        }
        # 目标形状顶点数
        self.target_vertices = {
            "橙色": 6,
            "浅橙色": 3,
            "绿色": 4,
            "粉色": 4,
        }
        # 英文形状名
        self.shape_names_en = {3: "Triangle", 4: "Quadrangle", 6: "Hexagon"}

        self.min_area_abs = min_area_abs  # 绝对最小面积（像素）
        self.debug = debug

        # 形态学核（保持小核，避免焊死）
        self.kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))

        # 分离参数（可微调）
        self.dist_thresh_ratio = 0.05     # 分水岭前景阈值
        self.sep_strength = 0.02          # 凹点分割强度（本版未用）
        self.min_marker_area = 20         # 分水岭标记最小面积
        self.force_erode_iter = 2         # force_separation 腐蚀迭代次数

    def detect(self, bgr_image):
        """输入 BGR 图像，返回标注了颜色+形状标签的 BGR 图像"""
        img = bgr_image.copy()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]
        # 根据图像尺寸缩放绝对面积阈值（也可以直接使用 self.min_area_abs）
        min_area = self.min_area_abs * (img_w * img_h) / (800 * 600)

        result_img = img.copy()

        for name, (lower, upper) in self.color_types.items():
            # 1. 颜色掩膜
            mask = cv.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel_close)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel_open)

            # 2. 三步分离流水线：腐蚀断开 → 形状恢复 → 分水岭细化
            separated = self._force_separation(mask, iterations=self.force_erode_iter)
            separated = self._restore_shapes(mask, separated)
            separated = self._enhanced_watershed(separated,
                                                 self.dist_thresh_ratio,
                                                 self.min_marker_area)

            # 3. 提取轮廓
            contours, _ = cv.findContours(separated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            en_color = self.color_name_en.get(name, name)
            target_vert = self.target_vertices.get(name, 4)
            en_shape = self.shape_names_en.get(target_vert, f"{target_vert}gon")

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

                # 画黑色质心点
                cv.circle(result_img, (cx, cy), 3, (0, 0, 0), -1)
                # 英文标签（黑色，小字号）
                text = f"{en_color} {en_shape}"
                cv.putText(result_img, text, (cx + 6, cy - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result_img

    # ---------- 内部辅助函数 ----------
    def _force_separation(self, mask, kernel_size=3, iterations=2):
        """腐蚀 + 距离变换 + 膨胀，强制断开粘连区域"""
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv.erode(mask, kernel, iterations=iterations)
        dist = cv.distanceTransform(eroded, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist, 0.2 * dist.max(), 255, cv.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        final_mask = cv.dilate(sure_fg, kernel, iterations=iterations)
        return final_mask

    def _restore_shapes(self, original_mask, separated_mask):
        """利用原始 mask 恢复腐蚀后丢失的形状，同时保持分离"""
        contours, _ = cv.findContours(separated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(original_mask)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            crop_orig = original_mask[y:y + h, x:x + w]
            crop_sep = separated_mask[y:y + h, x:x + w]
            # 局部膨胀，确保覆盖完整原始零件
            crop_sep = cv.dilate(crop_sep, np.ones((3, 3), np.uint8), iterations=2)
            restored = cv.bitwise_and(crop_orig, crop_sep)
            final_mask[y:y + h, x:x + w] = cv.bitwise_or(final_mask[y:y + h, x:x + w], restored)
        return final_mask

    def _enhanced_watershed(self, mask, dist_thresh_ratio=0.05, min_marker_area=20):
        """迭代腐蚀前景标记，强制分裂后用作分水岭种子"""
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

        # 去除细小标记
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