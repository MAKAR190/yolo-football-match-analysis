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
        self.reassign_margin_boost = 0.12
        self.min_segmented_bootstrap_samples = 4
        self.min_bootstrap_samples_total = 8
        self.bootstrap_refit_frames = 80
        self.bootstrap_attempts = 0
        self.max_bootstrap_attempts = 3
        self.bootstrap_quality_threshold = 0.45
        self.team_fit_confidence = 0.0
        self.team_color_families = {}
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
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        y1 = int(0.28 * h)
        y2 = int(0.60 * h)
        x1 = int(0.25 * w)
        x2 = int(0.75 * w)
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        torso = crop[y1:y2, x1:x2]
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
            return None, 0.0
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_.reshape(torso_bgr.shape[0], torso_bgr.shape[1])
        hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        h, w = labels.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx_ref = 0.5 * (w - 1)
        cy_ref = 0.5 * (h - 1)
        dist_norm = np.sqrt((xx - cx_ref) ** 2 + (yy - cy_ref) ** 2)
        dist_norm = dist_norm / (np.max(dist_norm) + 1e-6)

        edge_band = np.zeros_like(labels, dtype=np.float64)
        edge_margin_x = max(1, int(0.12 * w))
        edge_margin_y = max(1, int(0.12 * h))
        edge_band[:, :edge_margin_x] = 1.0
        edge_band[:, w - edge_margin_x :] = 1.0
        edge_band[:edge_margin_y, :] = 1.0
        edge_band[h - edge_margin_y :, :] = 1.0

        best_cluster = None
        best_cluster_score = -1.0
        for cluster_idx in [0, 1]:
            cluster_mask = (labels == cluster_idx).astype(np.uint8)
            area = float(cluster_mask.sum())
            if area < 12:
                continue
            occupancy = area / float(h * w)
            if occupancy < 0.08:
                continue
            sat_vals = hsv[:, :, 1][cluster_mask == 1]
            if sat_vals.size == 0:
                continue
            mean_sat = float(np.mean(sat_vals)) / 255.0
            mean_chroma = float(np.mean(np.linalg.norm(torso_bgr[cluster_mask == 1].astype(np.float64), axis=1))) / 255.0
            center_score = 1.0 - float(np.mean(dist_norm[cluster_mask == 1]))
            edge_penalty = float(np.mean(edge_band[cluster_mask == 1]))
            score = (
                (1.5 * center_score)
                + (1.0 * mean_sat)
                + (0.7 * mean_chroma)
                + (0.4 * occupancy)
                - (0.9 * edge_penalty)
            )
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster = cluster_idx

        if best_cluster is None:
            return None, 0.0
        mask = (labels == best_cluster).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return None, 0.0
        best_label = None
        best_score = -1.0
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            cx, cy = centroids[lbl]
            dist = float(np.hypot(cx - cx_ref, cy - cy_ref))
            center_term = 1.0 - min(1.0, dist / (0.5 * max(h, w)))
            area_term = min(1.0, area / float(h * w))
            score = (0.7 * center_term) + (0.3 * area_term)
            if score > best_score:
                best_score = score
                best_label = lbl
        if best_label is None:
            return None, 0.0
        mask = (cc_labels == best_label).astype(np.uint8)
        coverage = float(mask.sum()) / float(h * w)
        confidence = max(0.0, min(1.0, (0.65 * max(0.0, best_cluster_score)) + (0.35 * coverage)))
        if mask.sum() < 12 or confidence < 0.30:
            return None, confidence
        return mask, confidence

    @staticmethod
    def _norm_hist(values, bins, value_range):
        hist = cv2.calcHist([values], [0], None, [bins], value_range).flatten()
        s = float(hist.sum())
        if s > 0:
            hist = hist / s
        return hist.astype(np.float64)

    @staticmethod
    def _hsv_from_bgr_triplet(bgr):
        patch = np.array([[list(map(float, bgr))]], dtype=np.uint8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
        return float(hsv[0]), float(hsv[1]), float(hsv[2])

    @staticmethod
    def _color_family_from_bgr(bgr):
        h, s, v = TeamAssigner._hsv_from_bgr_triplet(bgr)
        if s < 35 and v > 145:
            return "white_like", 1.0 - min(1.0, s / 35.0)
        if s < 28 and v < 95:
            return "dark_like", 1.0
        if s < 45:
            return "neutral", 0.35

        if 85 <= h <= 130:
            return "blue_like", min(1.0, s / 255.0)
        if 35 <= h < 85:
            return "green_like", min(1.0, s / 255.0)
        if 0 <= h < 15 or 150 <= h <= 179:
            return "red_like", min(1.0, s / 255.0)
        if 15 <= h < 35:
            return "yellow_orange_like", min(1.0, s / 255.0)
        return "other", 0.3

    def _hard_color_gate_team(self, mean_bgr):
        if mean_bgr is None or len(self.team_color_families) != 2:
            return None
        fam_player, conf = self._color_family_from_bgr(mean_bgr)
        if conf < 0.72:
            return None

        matched_teams = [
            team_id
            for team_id, fam in self.team_color_families.items()
            if fam == fam_player
        ]
        if len(matched_teams) == 1:
            return int(matched_teams[0])
        return None

    def _extract_player_feature(self, frame, bbox):
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None, None, {"quality": "none", "confidence": 0.0}
        torso = self._extract_torso(crop)
        if torso is None:
            return None, None, {"quality": "none", "confidence": 0.0}

        mask, seg_conf = self._segment_jersey_pixels(torso)
        if mask is None:
            feat, mean_bgr = self._extract_fallback_feature_from_torso(torso)
            return feat, mean_bgr, {"quality": "fallback", "confidence": float(seg_conf)}

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
        return feat, mean_bgr, {"quality": "segmented", "confidence": float(seg_conf)}

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
        qualities = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            feat, mean_bgr, meta = self._extract_player_feature(frame, bbox)
            if feat is None:
                continue
            features.append(feat)
            mean_bgrs.append(mean_bgr)
            qualities.append(meta)

        if len(features) < 2:
            return

        segmented_idx = [
            i
            for i, q in enumerate(qualities)
            if q.get("quality") == "segmented"
            and float(q.get("confidence", 0.0)) >= self.bootstrap_quality_threshold
        ]
        if len(segmented_idx) >= self.min_segmented_bootstrap_samples:
            fit_idx = segmented_idx
        elif len(features) >= self.min_bootstrap_samples_total:
            fit_idx = list(range(len(features)))
        else:
            return

        X = np.asarray([features[i] for i in fit_idx], dtype=np.float64)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(X)
        self.bootstrap_attempts += 1
        center_dist = float(
            np.linalg.norm(self.kmeans.cluster_centers_[0] - self.kmeans.cluster_centers_[1])
        )
        segmented_ratio = (
            float(len(segmented_idx)) / float(len(features)) if len(features) > 0 else 0.0
        )
        self.team_fit_confidence = max(
            0.0,
            min(1.0, (0.45 * segmented_ratio) + (0.55 * min(1.0, center_dist / 1.2))),
        )

        mean_bgrs = np.asarray([mean_bgrs[i] for i in fit_idx], dtype=np.float64)
        for team_idx in [0, 1]:
            idx = np.where(labels == team_idx)[0]
            if len(idx) == 0:
                color = (255, 0, 255)
            else:
                color = tuple(map(int, np.mean(mean_bgrs[idx], axis=0).tolist()))
            self.team_colors[team_idx + 1] = color
        self.team_color_families = {
            team_id: self._color_family_from_bgr(color)[0]
            for team_id, color in self.team_colors.items()
        }
        if (
            len(self.team_color_families) == 2
            and self.team_color_families.get(1) == self.team_color_families.get(2)
        ):
            # Disable hard gate if both learned teams look like same family.
            self.team_color_families = {}

    def should_retry_team_fit(self, frame_idx):
        return (
            self.kmeans is not None
            and frame_idx <= self.bootstrap_refit_frames
            and self.bootstrap_attempts < self.max_bootstrap_attempts
            and self.team_fit_confidence < 0.35
        )

    def get_player_team(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            return None

        feat, mean_bgr, meta = self._extract_player_feature(frame, player_bbox)
        if feat is None:
            return self.player_team_dict.get(player_id)

        prev_team = self.player_team_dict.get(player_id)
        quality = meta.get("quality", "none")
        conf = float(meta.get("confidence", 0.0))
        gated_team = self._hard_color_gate_team(mean_bgr)
        if gated_team is not None:
            if prev_team is not None and prev_team != gated_team:
                self.player_team_dict[player_id] = gated_team
            elif prev_team is None:
                self.player_team_dict[player_id] = gated_team
            return gated_team

        samples = self.player_feature_samples.setdefault(player_id, [])
        sample_weight = 1.0 if quality == "segmented" else 0.45
        samples.append((feat.astype(np.float64), sample_weight))
        if len(samples) > self.max_samples_per_player:
            samples.pop(0)
        if len(samples) < self.min_samples_per_player:
            return prev_team

        feats = np.asarray([s[0] for s in samples], dtype=np.float64)
        weights = np.asarray([s[1] for s in samples], dtype=np.float64)
        if np.sum(weights) <= 0:
            return None
        mean_feat = np.average(feats, axis=0, weights=weights).reshape(1, -1)

        dists = np.linalg.norm(self.kmeans.cluster_centers_ - mean_feat, axis=1)
        d_sorted = np.sort(dists)
        margin = float(d_sorted[1] - d_sorted[0]) if len(d_sorted) >= 2 else 0.0
        best_team = int(np.argmin(dists)) + 1
        if margin < self.assignment_margin:
            return prev_team

        if (
            prev_team is not None
            and best_team != prev_team
            and (quality != "segmented" or conf < 0.45 or margin < (self.assignment_margin + self.reassign_margin_boost))
        ):
            return prev_team

        self.player_team_dict[player_id] = best_team
        return best_team
