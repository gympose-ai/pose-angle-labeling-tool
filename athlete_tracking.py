"""Conservative, single-target LK optical flow; never reacquire another person."""
import cv2
import numpy as np


class AthleteTracker:
    def __init__(self, frame, normalized_bbox):
        box = np.asarray(normalized_bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1).any():
            raise ValueError("Takip için ilk karede geçerli bir atlet kutusu seçin.")
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.scale = min(1.0, 960 / max(w, h))
        self.gray = self._gray(frame)
        x1, y1, x2, y2 = box * [w, h, w, h] * self.scale
        if x2 - x1 < 12 or y2 - y1 < 12:
            raise ValueError("Takip kutusu çok küçük; atletin tüm vücudunu seçin.")
        self.corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        self.points = self._features(self.gray)
        self.lost = self.points is None or len(self.points) < 8
        self.reason = "Başlangıç bölgesinde yeterli takip noktası bulunamadı." if self.lost else ""

    def _gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return gray

    def _features(self, gray):
        lo, hi = self.corners.min(axis=0), self.corners.max(axis=0)
        # Favor the subject's central area over background near the box edges.
        margin = (hi - lo) * 0.15
        lo = np.maximum(np.ceil(lo + margin).astype(int), 0)
        hi = np.minimum(np.floor(hi - margin).astype(int), [gray.shape[1], gray.shape[0]])
        mask = np.zeros_like(gray)
        mask[lo[1]:hi[1], lo[0]:hi[0]] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.015,
                                      minDistance=4, mask=mask, blockSize=5)

    @property
    def bbox(self):
        if self.lost:
            return None
        lo = np.maximum(self.corners.min(axis=0) / self.scale, [0, 0])
        hi = np.minimum(self.corners.max(axis=0) / self.scale, self.size)
        return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]

    def _lose(self, reason):
        self.lost = True
        self.reason = reason
        return None

    def update(self, frame):
        if self.lost:
            return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            return self._lose("Video boyutu değişti.")
        gray = self._gray(frame)
        try:
            lk = dict(winSize=(31, 31), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **lk)
            if next_points is None:
                return self._lose("Takip noktaları kayboldu.")
            previous, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, next_points, None, **lk)
            if previous is None:
                return self._lose("Takip doğrulanamadı.")
            error = np.linalg.norm(previous - self.points, axis=2).ravel()
            valid = (status.ravel() == 1) & (reverse_status.ravel() == 1) & (error < 1.5)
            if valid.sum() < 8 or valid.mean() < 0.45:
                return self._lose("Hareket veya örtülme nedeniyle takip güvenilir değil.")
            src, dst = self.points[valid].reshape(-1, 2), next_points[valid].reshape(-1, 2)
            transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                           ransacReprojThreshold=2.5)
            if transform is None or inliers is None or inliers.sum() < 8 or inliers.mean() < 0.6:
                return self._lose("Atletin hareketi tutarlı izlenemedi.")
            scale = np.linalg.norm(transform[0, :2])
            if not 0.85 <= scale <= 1.18:
                return self._lose("Takip alanı beklenmedik şekilde değişti.")
            new_corners = cv2.transform(self.corners[None], transform)[0]
            shift = np.linalg.norm(new_corners.mean(axis=0) - self.corners.mean(axis=0))
            if shift > max(25, np.linalg.norm(np.ptp(self.corners, axis=0)) * 0.4):
                return self._lose("Atletin konumu doğrulanamadı.")
            self.corners = new_corners
            box = self.bbox
            if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                return self._lose("Atlet görüntü dışına çıktı.")
            self.gray = gray
            # Retain only geometrically verified tracks. Replenish within the
            # same target region when needed, never by global person detection.
            self.points = dst[inliers.ravel().astype(bool)].reshape(-1, 1, 2).astype(np.float32)
            if len(self.points) < 30:
                features = self._features(gray)
                if features is not None:
                    self.points = features
            return box
        except cv2.error:
            return self._lose("Optical flow takibi tamamlanamadı.")
