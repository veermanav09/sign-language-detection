import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class HandTrackerOpenCV:
    """
    Alternative hand tracking using OpenCV instead of MediaPipe.
    Compatible with Python 3.13 and ARM64 systems.
    """
    
    def __init__(self):
        """Initialize OpenCV-based hand tracking."""
        # Load pre-trained hand detection model
        self.hand_cascade = None
        self.face_cascade = None
        
        # Try to load OpenCV's built-in cascades
        try:
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
            if self.hand_cascade.empty():
                self.hand_cascade = None
        except:
            self.hand_cascade = None
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
        
        # Hand tracking state
        self.prev_hand_center = None
        self.hand_trail = []
        self.max_trail_length = 10
        
        # Feature extraction parameters
        self.feature_size = 47
        
        # Robustness/tuning parameters
        # Lower min_area to increase sensitivity to smaller/partial hands
        self.min_area = 2500
        self.max_area = 110000
        self.max_center_jump = 0.12
        self.ema_alpha = 0.4
        self.last_bbox = None
        
        # ROI gating (start disabled to avoid missing hands)
        self.use_center_roi = False
        self.roi_margin = 0.15
        
        # Skin thresholds (HSV + YCrCb)
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.skin_lower2 = np.array([170, 20, 70], dtype=np.uint8)
        self.skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self.cr_min, self.cr_max = 135, 180
        self.cb_min, self.cb_max = 85, 135
        
        self.last_calibrated_at = None
        
        # Background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
        
        # Debug
        self.debug = False
        
    def set_debug(self, enabled: bool):
        self.debug = bool(enabled)
    
    def set_center_roi(self, enabled: bool):
        self.use_center_roi = bool(enabled)
    
    def calibrate(self, frame_bgr: np.ndarray):
        try:
            norm = self._normalize_lighting(frame_bgr)
            h, w = norm.shape[:2]
            x1 = int(self.roi_margin * w)
            x2 = int((1 - self.roi_margin) * w)
            y1 = int(self.roi_margin * h)
            y2 = int((1 - self.roi_margin) * h)
            crop = norm[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            med = np.median(hsv.reshape(-1, 3), axis=0)
            h_med, s_med, v_med = med.astype(np.uint8)
            h_delta = 15
            s_min = max(10, int(s_med * 0.5))
            v_min = max(40, int(v_med * 0.5))
            self.skin_lower = np.array([max(0, int(h_med - h_delta)), s_min, v_min], dtype=np.uint8)
            self.skin_upper = np.array([min(179, int(h_med + h_delta)), 255, 255], dtype=np.uint8)
            cr = ycrcb[:, :, 1].reshape(-1)
            cb = ycrcb[:, :, 2].reshape(-1)
            self.cr_min = int(np.percentile(cr, 20))
            self.cr_max = int(np.percentile(cr, 95))
            self.cb_min = int(np.percentile(cb, 20))
            self.cb_max = int(np.percentile(cb, 95))
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
            self.last_calibrated_at = cv2.getTickCount() / cv2.getTickFrequency()
        except Exception:
            pass
    
    def _normalize_lighting(self, image_bgr: np.ndarray) -> np.ndarray:
        result = image_bgr.astype(np.float32)
        mean_per_channel = result.mean(axis=(0, 1), keepdims=True) + 1e-6
        result = result * (128.0 / mean_per_channel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    def _apply_center_roi(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_center_roi:
            return mask
        h, w = mask.shape[:2]
        x1 = int(self.roi_margin * w)
        x2 = int((1 - self.roi_margin) * w)
        y1 = int(self.roi_margin * h)
        y2 = int((1 - self.roi_margin) * h)
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(mask, roi_mask)
    
    def _skin_mask_hsv(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        mask2 = cv2.inRange(hsv, self.skin_lower2, self.skin_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def _skin_mask_ycrcb(self, image: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        mask = cv2.inRange(Cr, self.cr_min, self.cr_max)
        mask2 = cv2.inRange(Cb, self.cb_min, self.cb_max)
        mask = cv2.bitwise_and(mask, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges
    
    def _bg_mask(self, frame: np.ndarray) -> np.ndarray:
        fg = self.bg_sub.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return fg
    
    def _fuse_masks(self, hsv_mask: np.ndarray, ycrcb_mask: np.ndarray, motion: np.ndarray, fg: np.ndarray) -> np.ndarray:
        skin_strict = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        skin_union = cv2.bitwise_or(hsv_mask, ycrcb_mask)
        skin_motion = cv2.bitwise_and(skin_union, motion)
        skin_fg = cv2.bitwise_and(skin_union, fg)
        fused = cv2.bitwise_or(skin_motion, skin_fg)
        if cv2.countNonZero(fused) < 50:
            fused = skin_union
        kernel = np.ones((5, 5), np.uint8)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)
        return fused
    
    def _is_hand_shape(self, cnt: np.ndarray) -> bool:
        area = cv2.contourArea(cnt)
        if area < self.min_area or area > self.max_area:
            return False
        hull = cv2.convexHull(cnt, returnPoints=True)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        # Relax solidity threshold slightly to avoid dropping valid hands
        if solidity < 0.5:
            return False
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            # If we cannot compute defects, still allow based on area/solidity
            return True
        defects = cv2.convexityDefects(cnt, hull_indices)
        if defects is None:
            # Some valid hands may not yield clear defects; allow them
            return True
        finger_like = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            # Slightly lower depth threshold to count shallow gaps between fingers
            if depth > 5:
                finger_like += 1
        return 1 <= finger_like <= 8
    
    def _filter_and_extract(self, mask: np.ndarray, image: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = image.shape[:2]
        best = None
        best_score = -1
        for cnt in contours:
            if not self._is_hand_shape(cnt):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            score = bw * bh
            if score > best_score:
                best = (x, y, bw, bh)
                best_score = score
        if best is None:
            return None
        x, y, bw, bh = best
        if self.last_bbox is None:
            self.last_bbox = (x, y, bw, bh)
        else:
            lx, ly, lw, lh = self.last_bbox
            x = int(self.ema_alpha * x + (1 - self.ema_alpha) * lx)
            y = int(self.ema_alpha * y + (1 - self.ema_alpha) * ly)
            bw = int(self.ema_alpha * bw + (1 - self.ema_alpha) * lw)
            bh = int(self.ema_alpha * bh + (1 - self.ema_alpha) * lh)
            self.last_bbox = (x, y, bw, bh)
        cx = x + bw / 2
        cy = y + bh / 2
        if self.prev_hand_center is not None:
            px, py = self.prev_hand_center
            jump = np.hypot(cx - px, cy - py)
            gate = self.max_center_jump * min(w, h)
            if jump > gate:
                cx, cy = px, py
                x = int(max(0, min(w - 1, cx - bw / 2)))
                y = int(max(0, min(h - 1, cy - bh / 2)))
        self.prev_hand_center = (cx, cy)
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
        cv2.putText(image, "Hand", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+bh, x:x+bw]
        return self._extract_hand_features_opencv(roi, x, y, bw, bh, w, h)
    
    def detect_hands_opencv(self, image: np.ndarray) -> List[np.ndarray]:
        frame = self._normalize_lighting(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_features = []
        try:
            hsv_mask = self._skin_mask_hsv(frame)
            y_mask = self._skin_mask_ycrcb(frame)
            motion = self._motion_mask(gray)
            fg = self._bg_mask(frame)
            hsv_mask = self._apply_center_roi(hsv_mask)
            y_mask = self._apply_center_roi(y_mask)
            motion = self._apply_center_roi(motion)
            fg = self._apply_center_roi(fg)
            fused = self._fuse_masks(hsv_mask, y_mask, motion, fg)
            if self.debug:
                cv2.imshow('dbg_skin_hsv', hsv_mask)
                cv2.imshow('dbg_skin_ycrcb', y_mask)
                cv2.imshow('dbg_fused', fused)
            features = self._filter_and_extract(fused, frame)
            if features is not None:
                hand_features.append(features)
        except Exception:
            pass
        if not hand_features and self.hand_cascade:
            try:
                hands = self.hand_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in hands:
                    roi = gray[y:y+h, x:x+w]
                    features = self._extract_hand_features_opencv(roi, x, y, w, h, frame.shape[1], frame.shape[0])
                    if features is not None:
                        hand_features.append(features)
                        break
            except Exception:
                pass
        return hand_features
    
    def _extract_hand_features_opencv(self, hand_roi: np.ndarray, x: int, y: int, w: int, h: int, img_width: int, img_height: int) -> Optional[np.ndarray]:
        try:
            if hand_roi is None or hand_roi.size == 0:
                return None
            # Ensure ROI has a consistent minimal size for feature stability
            hand_roi = cv2.resize(hand_roi, (21, 21))
            features = []
            x_norm = max(0.0, min(1.0, x / max(1, img_width)))
            y_norm = max(0.0, min(1.0, y / max(1, img_height)))
            w_norm = max(1.0 / max(1, img_width), min(1.0, w / max(1, img_width)))
            h_norm = max(1.0 / max(1, img_height), min(1.0, h / max(1, img_height)))
            for i in range(21):
                angle = (i / 21) * 2 * np.pi
                radius = 0.1 + 0.05 * (i % 3)
                landmark_x = x_norm + w_norm * (0.5 + radius * np.cos(angle))
                landmark_y = y_norm + h_norm * (0.5 + radius * np.sin(angle))
                features.extend([float(landmark_x), float(landmark_y)])
            # Append simple geometric features: width, height, aspect ratio, center x, center y
            roi_features = [
                float(w_norm),
                float(h_norm),
                float(w_norm / max(h_norm, 1e-6)),
                float(x_norm + w_norm / 2),
                float(y_norm + h_norm / 2),
            ]
            features.extend(roi_features)
            if len(features) < 47:
                features.extend([0.0] * (47 - len(features)))
            elif len(features) > 47:
                features = features[:47]
            return np.array(features, dtype=np.float32)
        except Exception:
            return None
    
    def draw_overlays(self, image: np.ndarray):
        try:
            h, w = image.shape[:2]
            if self.use_center_roi:
                x1 = int(self.roi_margin * w)
                x2 = int((1 - self.roi_margin) * w)
                y1 = int(self.roi_margin * h)
                y2 = int((1 - self.roi_margin) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (50, 200, 50), 1)
                cv2.putText(image, "ROI ON", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 1)
            if self.last_calibrated_at:
                cv2.putText(image, "Calibrated", (10, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1)
        except Exception:
            pass
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        hand_features = self.detect_hands_opencv(image)
        cv2.putText(image, f"Hands: {len(hand_features)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if hand_features:
            cv2.putText(image, "OpenCV Detection", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.draw_overlays(image)
        return image, hand_features
    
    def extract_hand_features(self, landmarks) -> np.ndarray:
        if landmarks is None:
            return np.zeros(47)
        return landmarks
    
    def calculate_hand_angles(self, landmarks) -> List[float]:
        if landmarks is None or len(landmarks) < 42:
            return [0.0] * 5
        angles = []
        for i in range(5):
            if i * 8 + 1 < len(landmarks):
                angle = (landmarks[i * 8] + landmarks[i * 8 + 1]) * 180
                angles.append(angle)
            else:
                angles.append(0.0)
        return angles
    
    def get_hand_bbox(self, landmarks) -> Optional[Tuple[int, int, int, int]]:
        if landmarks is None or len(landmarks) < 42:
            return None
        x = int(landmarks[42] * 640)
        y = int(landmarks[43] * 480)
        w = int(landmarks[44] * 640)
        h = int(landmarks[45] * 480)
        return (x - w//2, y - h//2, w, h)
    
    def is_hand_visible(self, landmarks) -> bool:
        if landmarks is None:
            return False
        if len(landmarks) < 47:
            return False
        for i in range(42):
            if not (0 <= landmarks[i] <= 1):
                return False
        return True
    
    def release(self):
        pass

# Example usage
if __name__ == "__main__":
    tracker = HandTrackerOpenCV()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("OpenCV Hand Tracking Demo")
    print("Press 'q' to quit, 'd' toggle debug")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, hand_features = tracker.detect_hands(frame)
            hand_count = len(hand_features)
            cv2.putText(processed_frame, f"Hands: {hand_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('OpenCV Hand Tracking', processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                tracker.set_debug(not tracker.debug)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
