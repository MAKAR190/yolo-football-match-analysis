import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_feature_samples = {}
        self.kmeans = None
        self.min_samples_per_player = 6
        self.max_samples_per_player = 12
        self.assignment_margin = 0.08
        self.unknown_color = (128, 128, 128)

    @staticmethod
    def _safe_crop(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        return crop

    @staticmethod
    def _extract_torso(crop):
        torso = crop[: max(1, crop.shape[0] // 2), :]
        tw = torso.shape[1]
        xl = int(0.2 * tw)
        xr = int(0.8 * tw)
        if xr > xl:
            torso = torso[:, xl:xr]
        return torso if torso.size > 0 else None

    @staticmethod
    def get_torso_roi_bbox(frame_shape, bbox):
        """
        Return torso ROI in full-frame coordinates used for team color assignment.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        # Lower + smaller torso window (more jersey-focused, less head/background):
        # vertical slice: 28%..60% of bbox height
        bh = y2 - y1
        torso_y1 = y1 + int(0.28 * bh)
        torso_y2 = y1 + int(0.60 * bh)
        torso_y1 = max(y1, min(torso_y1, y2 - 1))
        torso_y2 = max(torso_y1 + 1, min(torso_y2, y2))

        # Center 50% width
        bw = x2 - x1
        torso_x1 = x1 + int(0.25 * bw)
        torso_x2 = x1 + int(0.75 * bw)
        torso_x1 = max(x1, min(torso_x1, x2 - 1))
        torso_x2 = max(torso_x1 + 1, min(torso_x2, x2))

        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            return None
        return [int(torso_x1), int(torso_y1), int(torso_x2), int(torso_y2)]

    def _segment_jersey_pixels(self, torso_bgr):
        pixels = torso_bgr.reshape(-1, 3)
        if pixels.shape[0] < 2:
            return None
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_.reshape(torso_bgr.shape[0], torso_bgr.shape[1])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster
        mask = (labels == player_cluster).astype(np.uint8)
        # Morphological cleanup to remove noise and fill small holes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Keep connected component closest to torso center (usually jersey region).
        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return None
        h, w = mask.shape[:2]
        cx_ref = 0.5 * w
        cy_ref = 0.5 * h
        best_label = None
        best_score = float("inf")
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            cx, cy = centroids[lbl]
            dist = float(np.hypot(cx - cx_ref, cy - cy_ref))
            score = dist - 0.02 * area
            if score < best_score:
                best_score = score
                best_label = lbl
        if best_label is None:
            return None
        mask = (cc_labels == best_label).astype(np.uint8)
        if mask.sum() < 12:
            return None
        return mask

    @staticmethod
    def _norm_hist(values, bins, value_range):
        hist = cv2.calcHist([values], [0], None, [bins], value_range).flatten()
        s = float(hist.sum())
        if s > 0:
            hist = hist / s
        return hist.astype(np.float64)

    def _extract_player_feature(self, frame, bbox):
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None, None
        torso = self._extract_torso(crop)
        if torso is None:
            return None, None

        mask = self._segment_jersey_pixels(torso)
        if mask is None:
            return None, None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

        jersey_pixels_bgr = torso[mask == 1]
        if jersey_pixels_bgr.size == 0:
            return None, None
        mean_bgr = np.mean(jersey_pixels_bgr, axis=0).astype(np.float64)

        h_vals = hsv[:, :, 0][mask == 1]
        s_vals = hsv[:, :, 1][mask == 1]
        l_vals = lab[:, :, 0][mask == 1]
        a_vals = lab[:, :, 1][mask == 1]
        b_vals = lab[:, :, 2][mask == 1]

        feat = np.concatenate(
            [
                self._norm_hist(h_vals, bins=16, value_range=[0, 180]),
                self._norm_hist(s_vals, bins=16, value_range=[0, 256]),
                self._norm_hist(l_vals, bins=16, value_range=[0, 256]),
                self._norm_hist(a_vals, bins=16, value_range=[0, 256]),
                self._norm_hist(b_vals, bins=16, value_range=[0, 256]),
            ],
            axis=0,
        ).astype(np.float64)

        return feat, mean_bgr

    @staticmethod
    def _extract_fallback_feature_from_torso(torso):
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        mean_bgr = np.mean(torso.reshape(-1, 3), axis=0).astype(np.float64)

        h_vals = hsv[:, :, 0].reshape(-1)
        s_vals = hsv[:, :, 1].reshape(-1)
        l_vals = lab[:, :, 0].reshape(-1)
        a_vals = lab[:, :, 1].reshape(-1)
        b_vals = lab[:, :, 2].reshape(-1)

        def norm_hist(vals, bins, value_range):
            hist = cv2.calcHist([vals], [0], None, [bins], value_range).flatten()
            s = float(hist.sum())
            if s > 0:
                hist = hist / s
            return hist.astype(np.float64)

        feat = np.concatenate(
            [
                norm_hist(h_vals, bins=16, value_range=[0, 180]),
                norm_hist(s_vals, bins=16, value_range=[0, 256]),
                norm_hist(l_vals, bins=16, value_range=[0, 256]),
                norm_hist(a_vals, bins=16, value_range=[0, 256]),
                norm_hist(b_vals, bins=16, value_range=[0, 256]),
            ],
            axis=0,
        ).astype(np.float64)
        return feat, mean_bgr

    def assign_team_color(self, frame, player_detections):
        features = []
        mean_bgrs = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            feat, mean_bgr = self._extract_player_feature(frame, bbox)
            if feat is None:
                crop = self._safe_crop(frame, bbox)
                torso = self._extract_torso(crop) if crop is not None else None
                if torso is not None and torso.size > 0:
                    feat, mean_bgr = self._extract_fallback_feature_from_torso(torso)
            if feat is None:
                continue
            features.append(feat)
            mean_bgrs.append(mean_bgr)

        if len(features) < 2:
            return

        X = np.asarray(features, dtype=np.float64)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(X)

        mean_bgrs = np.asarray(mean_bgrs, dtype=np.float64)
        for team_idx in [0, 1]:
            idx = np.where(labels == team_idx)[0]
            if len(idx) == 0:
                color = (255, 0, 255)
            else:
                color = tuple(map(int, np.mean(mean_bgrs[idx], axis=0).tolist()))
            self.team_colors[team_idx + 1] = color

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return None

        feat, _ = self._extract_player_feature(frame, player_bbox)
        if feat is None:
            crop = self._safe_crop(frame, player_bbox)
            torso = self._extract_torso(crop) if crop is not None else None
            if torso is None or torso.size == 0:
                return None
            feat, _ = self._extract_fallback_feature_from_torso(torso)
            if feat is None:
                return None

        samples = self.player_feature_samples.setdefault(player_id, [])
        samples.append(feat.astype(np.float64))
        if len(samples) > self.max_samples_per_player:
            samples.pop(0)
        if len(samples) < self.min_samples_per_player:
            return None

        mean_feat = np.mean(np.asarray(samples, dtype=np.float64), axis=0, keepdims=True)
        dists = np.linalg.norm(self.kmeans.cluster_centers_ - mean_feat, axis=1)
        d_sorted = np.sort(dists)
        margin = float(d_sorted[1] - d_sorted[0]) if len(d_sorted) >= 2 else 0.0
        if margin < self.assignment_margin:
            return None

        team_id = int(np.argmin(dists)) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
