# /// script
# requires-python = ">=3.11"
# # (Neon Player venv detected as Python 3.12 on this machine)
# dependencies = [
#   "insightface>=0.7.3",
#   "onnxruntime>=1.16.0",
#   "opencv-python>=4.8.0",
#   "numpy>=1.24.0",
#   "pl-neon-recording>=0.1.13",
# ]
# ///
"""
RetinaFace Face Mapper Plugin for Neon Player
=============================================
Mirrors the Pupil Cloud "Face Mapper Enrichment" locally.

Outputs (written to <recording>/.neon_player/cache/RetinaFaceFaceMapper/):
  face_positions.csv      – per-frame bounding-boxes + 5 landmark points
  gaze_on_face.csv        – per-gaze-sample boolean "gaze on face"
  fixations_on_face.csv   – per-fixation boolean "fixation on face" with start/end timestamps

Model: RetinaFace-R50  (downloaded automatically via insightface on first run)

Installation
------------
Drop this file into:
  $HOME/Pupil Labs/Neon Player/plugins/

Neon Player will detect the PEP-723 dependencies above and install them
automatically the first time it starts with this plugin present.
"""

from __future__ import annotations

import csv
import logging
import os
import pathlib
import typing as T

import cv2
import numpy as np

from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from qt_property_widgets.utilities import property_params
from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLUGIN_NAME = "RetinaFaceFaceMapper"
FACE_POSITIONS_FILENAME = "face_positions.csv"
GAZE_ON_FACE_FILENAME = "gaze_on_face.csv"
FIXATIONS_ON_FACE_FILENAME = "fixations_on_face.csv"

FACE_POSITIONS_FIELDNAMES = [
    "recording id",
    "timestamp [ns]",
    "p1 x [px]",
    "p1 y [px]",
    "p2 x [px]",
    "p2 y [px]",
    "eye left x [px]",
    "eye left y [px]",
    "eye right x [px]",
    "eye right y [px]",
    "nose x [px]",
    "nose y [px]",
    "mouth left x [px]",
    "mouth left y [px]",
    "mouth right x [px]",
    "mouth right y [px]",
    "confidence",
]

GAZE_ON_FACE_FIELDNAMES = [
    "recording id",
    "timestamp [ns]",
    "gaze x [px]",
    "gaze y [px]",
    "gaze on face",
]

FIXATIONS_ON_FACE_FIELDNAMES = [
    "recording id",
    "fixation id",
    "start timestamp [ns]",
    "end timestamp [ns]",
    "fixation on face",
]

# RetinaFace landmark order: left_eye, right_eye, nose, mouth_left, mouth_right
_LM_NAMES = [
    ("eye left x [px]", "eye left y [px]"),
    ("eye right x [px]", "eye right y [px]"),
    ("nose x [px]", "nose y [px]"),
    ("mouth left x [px]", "mouth left y [px]"),
    ("mouth right x [px]", "mouth right y [px]"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_detector(det_size: tuple[int, int] = (640, 640)):
    """Load insightface RetinaFace-R50 model (downloads automatically)."""
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",           # insightface bundle that includes RetinaFace-R50
        allowed_modules=["detection"],  # detection only – no recognition
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def _gaze_on_faces(
    gaze_x: float,
    gaze_y: float,
    faces: list[dict],
) -> bool:
    """Return True if the gaze point falls inside any detected face bounding box."""
    for face in faces:
        x1, y1, x2, y2 = face["p1_x"], face["p1_y"], face["p2_x"], face["p2_y"]
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return True
    return False


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class RetinaFaceFaceMapper(Plugin):
    """
    RetinaFace-R50 face detection for Neon Player.

    Produces the same CSV outputs as Pupil Cloud's Face Mapper Enrichment:
      • face_positions.csv     – bounding boxes & 5 facial landmarks per frame
      • gaze_on_face.csv       – whether each gaze sample falls on a face
      • fixations_on_face.csv  – whether each fixation falls on a face
    """

    # ------------------------------------------------------------------ init

    def __init__(self) -> None:
        super().__init__()
        self._confidence_threshold: float = 0.5
        self._det_size_idx: int = 1          # 0=320, 1=640 (default), 2=1280
        self._draw_overlay: bool = True
        self._data: dict = {}
        self._status: str = "Idle"

    # ---------------------------------------------------------------- properties

    @property
    @property_params(min=0.1, max=1.0, decimals=2)
    def confidence_threshold(self) -> float:
        """Minimum detection confidence (0.1–1.0)."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = float(value)

    @property
    @property_params(min=0, max=2, decimals=0)
    def detection_size(self) -> int:
        """Input resolution index: 0=320px, 1=640px (default), 2=1280px."""
        return self._det_size_idx

    @detection_size.setter
    def detection_size(self, value: int) -> None:
        self._det_size_idx = int(value)

    @property
    def draw_overlay(self) -> bool:
        """Draw face boxes & landmarks on the scene video overlay."""
        return self._draw_overlay

    @draw_overlay.setter
    def draw_overlay(self, value: bool) -> None:
        self._draw_overlay = bool(value)

    # --------------------------------------------------------------- actions

    @action
    def run_detection(self) -> None:
        """Detect faces across the whole recording (runs in background)."""
        job = self.job_manager.run_background_action(
            "RetinaFace Detection",
            "RetinaFaceFaceMapper._detect_all_frames",
        )
        job.finished.connect(self.load_all())

    # @action
    # def export_csvs(self) -> None:
    #     """Export face_positions.csv, gaze_on_face.csv, and fixations_on_face.csv to the cache directory."""
    #     if not self._results:
    #         QMessageBox.warning(
    #             None,
    #             "No Detections",
    #             "Run detection first before exporting.",
    #         )
    #         return
    #     job = self.job_manager.run_background_action(
    #         "Exporting CSVs",
    #         "RetinaFaceFaceMapper._export",
    #     )
    #     job.finished.connect(
    #         lambda: QMessageBox.information(
    #             None, "Export Complete", f"CSVs saved to:\n{self._cache_dir()}"
    #         )
    #     )

    # @action
    # def run_and_export(self) -> None:
    #     """Run detection then immediately export CSVs (convenience action)."""
    #     job = self.job_manager.run_background_action(
    #         "RetinaFace Detection + Export",
    #         "RetinaFaceFaceMapper._detect_all_frames",
    #     )
    #     job.finished.connect(self._export_sync)

    # --------------------------------------------------------- background tasks

    def _detect_all_frames(self) -> T.Generator[ProgressUpdate, None, None]:
        """
        Background generator: iterates all scene video frames, runs RetinaFace,
        stores results keyed by frame timestamp (ns).
        """
        det_sizes = [(320, 320), (640, 640), (1280, 1280)]
        det_size = det_sizes[min(self._det_size_idx, 2)]

        recording = self.recording
        rec_path = pathlib.Path(recording._rec_dir)
        logger.info(f"_detect_all_frames started")
        logger.info(f"Recording path: {rec_path}")
        logger.info(f"Files: {[f.name for f in sorted(rec_path.iterdir())]}")

        logger.info(f"Loading RetinaFace-R50 (det_size={det_size})…")
        detector = _load_detector(det_size)

        scene_video_path = self._find_scene_video(recording)
        if scene_video_path is None:
            logger.error("No scene video found.")
            return

        cap = cv2.VideoCapture(str(scene_video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open {scene_video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps_ns = self._load_world_timestamps(recording)

        results = {}
        frame_idx = 0

        logger.info(f"Processing {total_frames} frames…")

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            ts_ns = (
                timestamps_ns[frame_idx]
                if frame_idx < len(timestamps_ns)
                else None
            )

            # insightface expects BGR uint8 numpy array
            faces_raw = detector.get(bgr)

            face_list: list[dict] = []
            for f in faces_raw:
                score = float(f.det_score)
                if score < self._confidence_threshold:
                    continue

                bbox = f.bbox.astype(int)          # [x1, y1, x2, y2]
                kps = f.kps.astype(int)             # (5, 2): lm in model order

                face_data: dict = {
                    "p1_x": int(bbox[0]),
                    "p1_y": int(bbox[1]),
                    "p2_x": int(bbox[2]),
                    "p2_y": int(bbox[3]),
                    "confidence": round(score, 4),
                }

                #logger.debug(f"p1=({face_data['p1_x']},{face_data['p1_y']}), p2=({face_data['p2_x']},{face_data['p2_y']}), confidence={face_data['confidence']}")
                #logger.debug(f)
                # Unpack 5 landmarks
                for (col_x, col_y), (lx, ly) in zip(_LM_NAMES, kps):
                    face_data[col_x] = int(lx)
                    face_data[col_y] = int(ly)

                face_list.append(face_data)

            if ts_ns is not None:
                results[ts_ns] = face_list
                #logger.debug(f"Frame {frame_idx}/{total_frames}, ts={ts_ns}: {len(face_list)} faces")

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                progress = frame_idx / max(total_frames, 1)
                yield ProgressUpdate(progress)

        cap.release()
        logger.info(
            f"Detection done. "
            f"{sum(len(v) for v in results.values())} total detections "
            f"across {len(results)} frames."
        )

        # save results to files in the cache directory
        logger.info("Starting export of results to CSV files…")
        self.export_all(results)

    def export_face_positions(self, results) -> None:
        """Background generator: writes face_positions.csv from self._results."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        face_positions_file = cache_dir / FACE_POSITIONS_FILENAME

        recording = self.recording
        recording_id = self._get_recording_id(recording)

        face_pos_path = cache_dir / FACE_POSITIONS_FILENAME
        with open(face_pos_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FACE_POSITIONS_FIELDNAMES)
            writer.writeheader()
            for ts_ns in sorted(results.keys()):
                for face in results[ts_ns]:
                    row: dict = {
                        "recording id": recording_id,
                        "timestamp [ns]": ts_ns,
                        "p1 x [px]": face["p1_x"],
                        "p1 y [px]": face["p1_y"],
                        "p2 x [px]": face["p2_x"],
                        "p2 y [px]": face["p2_y"],
                        "confidence": face["confidence"],
                    }
                    for col_x, col_y in _LM_NAMES:
                        row[col_x] = face.get(col_x, "")
                        row[col_y] = face.get(col_y, "")
                    writer.writerow(row)

        logger.info(f"Exported {FACE_POSITIONS_FILENAME} to {cache_dir}")

    def export_gaze_on_face(self, results) -> None:
        """Background generator: writes gaze_on_face.csv using self._results and gaze data."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        gaze_pos_path = cache_dir / GAZE_ON_FACE_FILENAME

        recording = self.recording
        recording_id = self._get_recording_id(recording)

        gaze_data = self._load_gaze(recording)  # list of (ts_ns, x_px, y_px)
        sorted_face_ts = sorted(results.keys())

        with open(gaze_pos_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=GAZE_ON_FACE_FIELDNAMES)
            writer.writeheader()
            for g_ts_ns, gx_px, gy_px in gaze_data:
                # Find nearest scene frame timestamp <= gaze timestamp
                frame_ts = self._nearest_frame_ts(g_ts_ns, sorted_face_ts)
                faces_at_frame = results.get(frame_ts, []) if frame_ts is not None else []
                on_face = _gaze_on_faces(gx_px, gy_px, faces_at_frame)
                writer.writerow(
                    {
                        "recording id": recording_id,
                        "timestamp [ns]": g_ts_ns,
                        "gaze x [px]": round(gx_px, 2),
                        "gaze y [px]": round(gy_px, 2),
                        "gaze on face": on_face,
                    }
                )

        logger.info(f"Exported {GAZE_ON_FACE_FILENAME} to {cache_dir}")

    def export_fixations_on_face(self, results) -> None:
        """Background generator: writes fixations_on_face.csv using self._results and fixation data."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        fix_path = cache_dir / FIXATIONS_ON_FACE_FILENAME

        recording = self.recording
        recording_id = self._get_recording_id(recording)

        fixation_data = self._load_fixations(recording)
        # (fixation_id, start_ns, end_ns, centroid_x, centroid_y)

        sorted_face_ts = sorted(results.keys())

        with open(fix_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIXATIONS_ON_FACE_FIELDNAMES)
            writer.writeheader()
            for fix_id, start_ns, end_ns, cx, cy in fixation_data:
                mid_ns = (start_ns + end_ns) // 2
                frame_ts = self._nearest_frame_ts(mid_ns, sorted_face_ts)
                faces_at_frame = results.get(frame_ts, []) if frame_ts is not None else []
                on_face = _gaze_on_faces(cx, cy, faces_at_frame)
                writer.writerow(
                    {
                        "recording id": recording_id,
                        "fixation id": fix_id,
                        "start timestamp [ns]": start_ns,
                        "end timestamp [ns]": end_ns,
                        "fixation on face": on_face,
                    }
                )
        logger.info(f"Exported {FIXATIONS_ON_FACE_FILENAME} to {cache_dir}")


    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self.load_all()

    def load_all(self) -> None:
        """Load detection results from cache and update status.
        Face information is stored in self._data['face_positions'] as a dict:
        {timestamp_ns: [face1_data, face2_data, ...], ...}
        where each face_data is a dict with keys:
        'p1_x', 'p1_y', 'p2_x', 'p2_y', 'confidence', and optional landmark keys.
        """
        cache_dir = self._cache_dir()
        face_pos_path = cache_dir / FACE_POSITIONS_FILENAME
        if not face_pos_path.exists():
            logger.error(f"No results found in cache at {cache_dir}")
            self._status = "No results found. Run detection first."
            return

        results = {}
        with open(face_pos_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts_ns = int(row["timestamp [ns]"])
                    face_data = {
                        "p1_x": int(row["p1 x [px]"]),
                        "p1_y": int(row["p1 y [px]"]),
                        "p2_x": int(row["p2 x [px]"]),
                        "p2_y": int(row["p2 y [px]"]),
                        "confidence": float(row["confidence"]),
                    }
                    for col_x, col_y in _LM_NAMES:
                        if row[col_x] and row[col_y]:
                            face_data[col_x] = int(row[col_x])
                            face_data[col_y] = int(row[col_y])
                    results.setdefault(ts_ns, []).append(face_data)
                    print(f"Loaded face at ts={ts_ns}: {face_data}")
                except Exception as exc:
                    logger.warning(f"Skipping malformed row: {row} ({exc})")

        self._data['face_positions'] = results
        self._on_detection_finished()

    def export_all(self, results) -> T.Generator[ProgressUpdate, None, None]:
        """Background generator: runs all export steps sequentially."""
        logger.info("export face positions…")
        self.export_face_positions(results)
        logger.info("export gaze on face…")
        self.export_gaze_on_face(results)
        logger.info("export fixations on face…")
        self.export_fixations_on_face(results)

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        #logger.info(f"render called with time_in_recording={time_in_recording}, scene_idx={scene_idx}")

        face_positions = self._data.get('face_positions', {})
        faces = face_positions.get(time_in_recording, [])   
        logger.debug(f"render: found {len(faces)} faces for time_in_recording={time_in_recording}")
        #logger.debug(f"render: found {len(faces)} faces for time_in_recording={time_in_recording}")



    # def _export(self) -> T.Generator[ProgressUpdate, None, None]:
    #     """Background generator: writes face_positions.csv and gaze_on_face.csv."""
    #     cache_dir = self._cache_dir()
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #     recording = self.recording

    #     recording_id = self._get_recording_id(recording)

    #     # --- face_positions.csv ---
    #     face_pos_path = cache_dir / FACE_POSITIONS_FILENAME
    #     with open(face_pos_path, "w", newline="", encoding="utf-8") as fh:
    #         writer = csv.DictWriter(fh, fieldnames=FACE_POSITIONS_FIELDNAMES)
    #         writer.writeheader()
    #         for ts_ns in sorted(self._results.keys()):
    #             for face in self._results[ts_ns]:
    #                 row: dict = {
    #                     "recording id": recording_id,
    #                     "timestamp [ns]": ts_ns,
    #                     "p1 x [px]": face["p1_x"],
    #                     "p1 y [px]": face["p1_y"],
    #                     "p2 x [px]": face["p2_x"],
    #                     "p2 y [px]": face["p2_y"],
    #                     "confidence": face["confidence"],
    #                 }
    #                 for col_x, col_y in _LM_NAMES:
    #                     row[col_x] = face.get(col_x, "")
    #                     row[col_y] = face.get(col_y, "")
    #                 writer.writerow(row)

    #     yield ProgressUpdate(0.5)

    #     # --- gaze_on_face.csv ---
    #     gaze_data = self._load_gaze(recording)  # list of (ts_ns, x_px, y_px)
    #     sorted_face_ts = sorted(self._results.keys())

    #     gaze_pos_path = cache_dir / GAZE_ON_FACE_FILENAME
    #     with open(gaze_pos_path, "w", newline="", encoding="utf-8") as fh:
    #         writer = csv.DictWriter(fh, fieldnames=GAZE_ON_FACE_FIELDNAMES)
    #         writer.writeheader()
    #         for g_ts_ns, gx_px, gy_px in gaze_data:
    #             # Find nearest scene frame timestamp <= gaze timestamp
    #             frame_ts = self._nearest_frame_ts(g_ts_ns, sorted_face_ts)
    #             faces_at_frame = self._results.get(frame_ts, []) if frame_ts is not None else []
    #             on_face = _gaze_on_faces(gx_px, gy_px, faces_at_frame)
    #             writer.writerow(
    #                 {
    #                     "recording id": recording_id,
    #                     "timestamp [ns]": g_ts_ns,
    #                     "gaze x [px]": round(gx_px, 2),
    #                     "gaze y [px]": round(gy_px, 2),
    #                     "gaze on face": on_face,
    #                 }
    #             )

    #     yield ProgressUpdate(0.75)

    #     # --- fixations_on_face.csv ---
    #     # A fixation is "on face" if its gaze centroid (average of all gaze samples
    #     # during the fixation window) falls inside any face bounding box at the
    #     # nearest scene frame to the fixation's midpoint timestamp.
    #     fixation_data = self._load_fixations(recording)
    #     # (fixation_id, start_ns, end_ns, centroid_x, centroid_y)

    #     fix_path = cache_dir / FIXATIONS_ON_FACE_FILENAME
    #     with open(fix_path, "w", newline="", encoding="utf-8") as fh:
    #         writer = csv.DictWriter(fh, fieldnames=FIXATIONS_ON_FACE_FIELDNAMES)
    #         writer.writeheader()
    #         for fix_id, start_ns, end_ns, cx, cy in fixation_data:
    #             mid_ns = (start_ns + end_ns) // 2
    #             frame_ts = self._nearest_frame_ts(mid_ns, sorted_face_ts)
    #             faces_at_frame = self._results.get(frame_ts, []) if frame_ts is not None else []
    #             on_face = _gaze_on_faces(cx, cy, faces_at_frame)
    #             writer.writerow(
    #                 {
    #                     "recording id": recording_id,
    #                     "fixation id": fix_id,
    #                     "start timestamp [ns]": start_ns,
    #                     "end timestamp [ns]": end_ns,
    #                     "fixation on face": on_face,
    #                 }
    #             )

    #     yield ProgressUpdate(1.0)
    #     logger.info(f"Exported CSVs to {cache_dir}")

    # --------------------------------------------------------- private helpers

    def _cache_dir(self) -> pathlib.Path:
        recording = self.recording
        rec_path = pathlib.Path(recording._rec_dir)
        return rec_path / ".neon_player" / "cache" / PLUGIN_NAME

    def _on_detection_finished(self) -> None:
        n_frames = len(self._results)
        n_faces = sum(len(v) for v in self._results.values())
        self._status = f"Done – {n_faces} detections in {n_frames} frames"
        logger.info(f"{self._status}")

    def _export_sync(self) -> None:
        """Called after detection job finishes to chain export."""
        job = self.job_manager.run_background_action(
            "Exporting CSVs",
            "RetinaFaceFaceMapper._export",
        )
        job.finished.connect(
            lambda: QMessageBox.information(
                None,
                "RetinaFace Face Mapper",
                f"Detection & export complete!\n"
                f"Results saved to:\n{self._cache_dir()}",
            )
        )

    # ---------------------------------------------------------------- recording helpers

    @staticmethod
    def _find_scene_video(recording) -> pathlib.Path | None:
        rec_path = pathlib.Path(recording._rec_dir)
        # Neon native naming: "Neon Scene Camera v1 ps1.mp4"
        for mp4 in sorted(rec_path.glob("*.mp4")):
            name = mp4.name.lower()
            if "eye" not in name and "sensor module" not in name:
                return mp4
        return None

    @staticmethod
    def _load_world_timestamps(recording) -> list[int]:
        """
        Load scene camera timestamps from Neon's native binary .time files.
        Neon stores timestamps as int64 nanoseconds in little-endian binary.

        File: "Neon Scene Camera v1 ps1.time"
        Also checks .time_aux and .time_hw as fallbacks.
        """
        rec_path = pathlib.Path(recording._rec_dir)
        timestamps_ns: list[int] = []

        # Find the scene camera .time file
        time_candidates = []
        for f in sorted(rec_path.iterdir()):
            name = f.name.lower()
            if (
                f.suffix in (".time", ".time_aux", ".time_hw")
                and "scene" in name
                and "eye" not in name
            ):
                time_candidates.append(f)

        # Prefer .time over .time_aux/.time_hw
        time_candidates.sort(key=lambda p: (p.suffix != ".time", p.name))

        for time_file in time_candidates:
            try:
                raw = time_file.read_bytes()
                # Neon .time files: int64 nanosecond timestamps, little-endian
                arr = np.frombuffer(raw, dtype="<i8")
                timestamps_ns = arr.tolist()
                logger.info(
                    f"Loaded {len(timestamps_ns)} "
                    f"timestamps from {time_file.name}"
                )
                return timestamps_ns
            except Exception as exc:
                logger.warning(
                    f"Could not read {time_file.name}: {exc}"
                )

        # Log available files to help diagnose if nothing found
        logger.error(
            "No scene .time file found. "
            f"Files: {[f.name for f in sorted(rec_path.iterdir())]}"
        )
        return timestamps_ns

    @staticmethod
    def _load_gaze(recording) -> list[tuple[int, float, float]]:
        """
        Load gaze from Neon native binary files:
          gaze ps1.raw  – float32 pairs: (x_px, y_px) per sample
          gaze ps1.time – int64 nanosecond timestamps
        Falls back to gaze_200hz if ps1 not present.
        """
        rec_path = pathlib.Path(recording._rec_dir)
        result: list[tuple[int, float, float]] = []

        # Find gaze .raw and .time files (prefer ps1, fallback to 200hz)
        raw_file = time_file = None
        for candidate_raw, candidate_time in [
            ("gaze ps1.raw", "gaze ps1.time"),
            ("gaze_200hz.raw", "gaze_200hz.time"),
        ]:
            r = rec_path / candidate_raw
            t = rec_path / candidate_time
            if r.exists() and t.exists():
                raw_file, time_file = r, t
                break

        if raw_file is None:
            logger.warning("No gaze binary files found")
            return result

        try:
            timestamps = np.frombuffer(time_file.read_bytes(), dtype="<i8")
            # gaze .raw: float32 x, y pairs
            xy = np.frombuffer(raw_file.read_bytes(), dtype="<f4").reshape(-1, 2)
            for ts, (x, y) in zip(timestamps, xy):
                result.append((int(ts), float(x), float(y)))
            logger.info(f"Loaded {len(result)} gaze samples")
        except Exception as exc:
            logger.warning(f"Could not load gaze: {exc}")

        return result

    @staticmethod
    def _load_fixations(recording) -> list[tuple[int, int, int, float, float]]:
        """
        Load fixations from Neon native binary files:
          fixations ps1.raw  – structured: start_ns(i8), end_ns(i8), x(f4), y(f4)
          fixations ps1.time – int64 nanosecond timestamps (one per fixation)
        Returns list of (fixation_id, start_ns, end_ns, centroid_x, centroid_y).
        """
        rec_path = pathlib.Path(recording._rec_dir)
        result: list[tuple[int, int, int, float, float]] = []

        raw_file = rec_path / "fixations ps1.raw"
        time_file = rec_path / "fixations ps1.time"

        if not raw_file.exists() or not time_file.exists():
            logger.warning("No fixation binary files found")
            return result

        try:
            timestamps = np.frombuffer(time_file.read_bytes(), dtype="<i8")
            raw_bytes = raw_file.read_bytes()
            logger.info("fixations - loaded raw bytes of length {}".format(len(raw_bytes)))

            dtype = np.dtype([
                ("event_type", "int32"),
                ("start_timestamp_ns",       "int64"),
                ("end_timestamp_ns",         "int64"),
                ("start_gaze_x",             "float32"),
                ("start_gaze_y",             "float32"),
                ("end_gaze_x",               "float32"),
                ("end_gaze_y",               "float32"),
                ("mean_gaze_x",              "float32"),
                ("mean_gaze_y",              "float32"),
                ("amplitude_pixels",         "float32"),
                ("amplitude_angle_deg",      "float32"),
                ("mean_velocity",            "float32"),
                ("max_velocity",             "float32"),
                ])

            fixations = np.frombuffer(raw_bytes, dtype=dtype)
            print("fixations - loaded structured array of shape {}".format(fixations.shape))

            for i, (fix, ts) in enumerate(zip(fixations, timestamps)):
                result.append((
                    i + 1,
                    int(fix["start_timestamp_ns"]),
                    int(fix["end_timestamp_ns"]),
                    float(fix["mean_gaze_x"]),
                    float(fix["mean_gaze_y"]),
                ))
            logger.info(f"Loaded {len(result)} fixations")
        except Exception as exc:
            logger.exception(f"Could not load fixations: {exc}")

        return result

    @staticmethod
    def _get_recording_id(recording) -> str:
        try:
            return str(recording.unique_id)
        except AttributeError:
            try:
                return str(recording._rec_dir)
            except AttributeError:
                return "unknown"

    @staticmethod
    def _nearest_frame_ts(
        gaze_ts_ns: int,
        sorted_frame_ts: list[int],
    ) -> int | None:
        """Binary-search for the largest frame timestamp <= gaze_ts_ns."""
        if not sorted_frame_ts:
            return None
        lo, hi = 0, len(sorted_frame_ts) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_frame_ts[mid] <= gaze_ts_ns:
                result = sorted_frame_ts[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return result
