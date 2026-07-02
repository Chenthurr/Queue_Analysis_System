"""
vision.py
---------
The core CV pipeline. Runs on its own background thread so that FastAPI's
event loop is never blocked by frame reads or model inference.

Pipeline per frame:
  1. Read frame from video source
  2. YOLOv8 inference, filtered to class 0 ('person')
  3. Convert each box -> "feet point" (bottom-center)
  4. Feed feet points into CentroidTracker -> stable per-person IDs
  5. Point-in-polygon test each tracked centroid against the ROI
  6. Debounce: because the tracker persists an ID through a short
     disappearance, a briefly-occluded person keeps their last known
     position and ROI status -- so the raw count doesn't flicker.
  7. Smoothing: raw per-frame counts are pushed into a rolling buffer;
     the DISPLAYED count is the rounded moving average of that buffer.
     This is a second, independent layer of stability on top of the
     tracker's own occlusion tolerance -- it also absorbs one-off
     detection noise (e.g. a single frame where YOLO misses a person
     who was never occluded at all).
"""
import cv2
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO
from tracker import CentroidTracker


class QueueVision:
    def __init__(self, video_source, roi_points, model_path="yolov8n.pt",
                 minutes_per_person=2.0, smoothing_window=8, max_disappeared=15):
        self.video_source = video_source
        # ROI polygon vertices in (x, y) pixel coordinates, in order
        # (clockwise or counter-clockwise both work for pointPolygonTest).
        self.roi_points = np.array(roi_points, dtype=np.int32)

        self.model = YOLO(model_path)
        self.tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=90)
        self.minutes_per_person = minutes_per_person

        # Rolling buffer of raw per-frame counts -> moving-average smoothing.
        self.count_buffer = deque(maxlen=smoothing_window)

        self._lock = threading.Lock()
        self._latest_state = {
            "queue_count": 0,
            "raw_count": 0,
            "estimated_wait_minutes": 0.0,
            "tracked_people": 0,
            "timestamp": time.time(),
        }
        self._running = False
        self._thread = None
        self.latest_frame = None  # exposed for the optional MJPEG debug stream

    # ---------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------
    def _point_in_roi(self, point):
        """
        Point-in-polygon test via cv2.pointPolygonTest.

        Math: pointPolygonTest returns a signed value:
            > 0  -> point lies strictly INSIDE the polygon
            = 0  -> point lies exactly ON an edge
            < 0  -> point lies OUTSIDE the polygon
        With measureDist=False it just returns +1 / 0 / -1 (fast path,
        no distance-to-edge calculation needed since we only care about
        inside/outside, not "how far outside").

        We treat >= 0 as "inside" so a person standing exactly on the
        boundary line still counts as in queue.
        """
        result = cv2.pointPolygonTest(self.roi_points, (float(point[0]), float(point[1])), False)
        return result >= 0

    @staticmethod
    def _bottom_center(box_xyxy):
        """
        Why bottom-center instead of box center?
        A person's true floor position is best approximated by where their
        feet touch the ground -- i.e. the bottom edge of the bounding box.
        The box's vertical center shifts substantially with the person's
        pose and how much of their body YOLO captures (e.g. a box that
        only captures torso+head vs full body), which would make the
        geometric center an unstable, unreliable proxy for "which tile of
        the floor are they standing on." The x-center is a reasonable
        proxy for lateral position either way.
        """
        x1, y1, x2, y2 = box_xyxy
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        return (cx, cy)

    # ---------------------------------------------------------------
    # Main capture/inference loop (runs in a background thread)
    # ---------------------------------------------------------------
    def _run_loop(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_source}")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Loop the demo file so it behaves like a continuous feed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # classes=[0] -> COCO class 0 is 'person'. verbose=False keeps
            # stdout clean since this runs every frame.
            results = self.model(frame, classes=[0], verbose=False)[0]

            feet_points = []
            boxes_xyxy = []
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                boxes_xyxy.append(xyxy)
                feet_points.append(self._bottom_center(xyxy))

            # IMPORTANT: feed ALL detected feet-points to the tracker, not
            # just the ones currently inside the ROI. This gives people a
            # stable ID as they approach or leave the queue line, so their
            # in/out ROI transition is also tracked cleanly rather than
            # them "spawning" a new ID the instant they cross the line.
            tracked_objects = self.tracker.update(feet_points)

            # Determine ROI membership per TRACKED person (not per raw
            # detection). Because the tracker holds a person's last known
            # centroid during their grace period (see CentroidTracker.
            # max_disappeared), someone occluded for 1-2 frames keeps
            # their previous ROI status instead of vanishing from the count.
            raw_count = 0
            for object_id, centroid in tracked_objects.items():
                if self._point_in_roi(centroid):
                    raw_count += 1

            with self._lock:
                self.count_buffer.append(raw_count)
                # Smoothed count = rounded moving average of the last N
                # raw per-frame counts. This absorbs any remaining
                # single-frame noise the tracker's grace period doesn't
                # already cover.
                smoothed_count = round(sum(self.count_buffer) / len(self.count_buffer))
                wait_minutes = round(smoothed_count * self.minutes_per_person, 1)

                self._latest_state = {
                    "queue_count": smoothed_count,
                    "raw_count": raw_count,
                    "estimated_wait_minutes": wait_minutes,
                    "tracked_people": len(tracked_objects),
                    "timestamp": time.time(),
                }

            self._draw_debug(frame, boxes_xyxy, tracked_objects, smoothed_count)
            self.latest_frame = frame

            time.sleep(0.01)  # small yield so this thread doesn't peg a core

        cap.release()

    def _draw_debug(self, frame, boxes, tracked_objects, smoothed_count):
        """Overlay ROI, boxes, track IDs and the live count. Purely visual,
        used only by the optional /api/video-feed debug stream."""
        cv2.polylines(frame, [self.roi_points], isClosed=True, color=(0, 255, 255), thickness=2)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for object_id, centroid in tracked_objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            color = (0, 0, 255) if self._point_in_roi(centroid) else (180, 180, 180)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Queue: {smoothed_count}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # ---------------------------------------------------------------
    # Public control / access API (called from main.py)
    # ---------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_state(self):
        with self._lock:
            return dict(self._latest_state)
