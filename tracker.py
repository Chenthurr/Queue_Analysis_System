"""
tracker.py
----------
Lightweight centroid-based multi-object tracker.

WHY WE NEED THIS (instead of just using raw YOLO boxes per frame):
YOLO gives us bounding boxes independently on every single frame -- it has
no concept of "this is the same person as last frame." If we just counted
"boxes inside the ROI" per frame, a person who is occluded for one frame
(someone walks in front of them, or a momentary missed detection) would
cause the queue count to flicker down and back up instantly. That looks
broken on a dashboard.

The tracker assigns each detected person a persistent integer ID across
frames. That ID is what lets vision.py apply debounce/grace-period logic
PER PERSON, rather than per raw detection.

The matching strategy here is intentionally simple: greedy nearest-centroid
matching using a Euclidean distance matrix. This is the same core idea
behind the classic "OpenCV/pyimagesearch centroid tracker." We don't need
a Kalman filter or full SORT/DeepSORT here because queue scenes are
low-density and low-velocity (people standing mostly still). The class is
still isolated behind a clean update() API, so swapping in a heavier
tracker later wouldn't require touching vision.py's logic.
"""
import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=90):
        """
        max_disappeared: how many CONSECUTIVE frames an object can go
            undetected before we give up on it and remove its ID.
            This number is directly your "occlusion tolerance" -- at a
            typical ~15-20 fps inference rate, 15 frames is roughly
            a 1-second grace period.
        max_distance: max allowed pixel distance between an existing
            track's last centroid and a new detection for them to be
            considered the same physical person. Prevents a track from
            "jumping" to an unrelated person on the other side of frame.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()       # objectID -> last known centroid (x, y)
        self.disappeared = OrderedDict()   # objectID -> consecutive frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        """
        input_centroids: list of (x, y) points detected THIS frame
        (one per YOLO 'person' detection -- see vision.py for how these
        are computed as bottom-center / "feet" points).

        Returns the current dict of {objectID: centroid}, including
        objects that are in their grace period (temporarily undetected
        but not yet timed out).
        """
        # Case 1: nothing detected this frame -> age out every existing track
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(input_centroids)

        # Case 2: no existing tracks yet -> register every detection as new
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Pairwise Euclidean distance matrix:
            # rows = existing tracks, cols = new detections this frame.
            # D[i, j] = distance between existing track i and new detection j
            D = np.linalg.norm(
                object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2,
            )

            # Greedy assignment: for each row, find its smallest distance,
            # then process rows in order of "most confident match first"
            # so we don't let an ambiguous pair steal a detection that
            # clearly belongs to a more confident pair.
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue  # too far apart to plausibly be the same person
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            # Existing tracks that found no match this frame -> increment
            # their "missing" counter (this IS the occlusion tolerance).
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Detections that matched no existing track -> brand-new people
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
