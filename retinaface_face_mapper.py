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

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from qt_property_widgets.utilities import property_params, action_params
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QColor, QIcon, QPainter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLUGIN_NAME = "RetinaFaceFaceMapper"
FACE_POSITIONS_FILENAME = "face_positions.csv"
GAZE_ON_FACE_FILENAME = "gaze_on_face.csv"
FIXATIONS_ON_FACE_FILENAME = "fixations_on_face.csv"

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


    # action to run from command line
    @action
    def detect_and_export(self, destination: pathlib.Path = pathlib.Path()) -> T.Generator[ProgressUpdate, None, None]:
        print("detect_and_export called with destination:", destination)
        yield from self._detect_all_frames()
        print("detect_and_export call export() with destination:", destination)
        yield from self.export(destination)

    # --------------------------------------------------------------- actions

    @action
    def run_detection(self) -> None:
        """Detect faces across the whole recording (runs in background)."""
        job = self.job_manager.run_background_action(
            "RetinaFace Detection",
            "RetinaFaceFaceMapper._detect_all_frames",
        )
        job.finished.connect(self._load_all_from_cache)

    # --------------------------------------------------------- background tasks

    def _detect_all_frames(self) -> T.Generator[ProgressUpdate, None, None]:
        """
        Background generator: iterates all scene video frames, runs RetinaFace,
        stores results keyed by frame index.
        """
        det_sizes = [(320, 320), (640, 640), (1280, 1280)]
        det_size = det_sizes[min(self._det_size_idx, 2)]

        recording = self.recording
        recording_id = self._get_recording_id(recording)
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


        if self._data is not None:
            logger.info("Dumping current face position data before detection…")
            self._data = None

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
                    "scene_idx": frame_idx,
                    "ts_ns": ts_ns,
                    "p1_x": int(bbox[0]),
                    "p1_y": int(bbox[1]),
                    "p2_x": int(bbox[2]),
                    "p2_y": int(bbox[3]),
                    "confidence": round(score, 4),
                    "el_x": int(kps[0][0]),
                    "el_y": int(kps[0][1]),
                    "er_x": int(kps[1][0]),
                    "er_y": int(kps[1][1]),
                    "n_x": int(kps[2][0]),
                    "n_y": int(kps[2][1]),
                    "ml_x": int(kps[3][0]),
                    "ml_y": int(kps[3][1]),
                    "mr_x": int(kps[4][0]),
                    "mr_y": int(kps[4][1]),
                    "recording_id": recording_id,
                }

                # #logger.debug(f"p1=({face_data['p1_x']},{face_data['p1_y']}), p2=({face_data['p2_x']},{face_data['p2_y']}), confidence={face_data['confidence']}")
                # #logger.debug(f)
                # # Unpack 5 landmarks
                # for (col_x, col_y), (lx, ly) in zip(_LM_NAMES, kps):
                #     face_data[col_x] = int(lx)
                #     face_data[col_y] = int(ly)

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

        # save results, get other stuff
        self._data = {}
        self._data['face_positions'] = results
        self._data['gaze_on_face'] = self._find_all_gaze_on_face(results, self._load_gaze(recording))
        self._data['fixations_on_face'] = self._find_all_fixations_on_face(results, self._load_fixations(recording))
        self._data['ts_ns_list'] = timestamps_ns
        # save results to files in the cache directory
        logger.info("Starting export of results to cache directory…")
        self.export(self._cache_dir())


    def _export_face_positions(self, face_positions, destination: pathlib.Path = pathlib.Path()) -> None:
        """Background generator: writes face_positions.csv from self._results."""
        destination.mkdir(parents=True, exist_ok=True)
        face_positions_file = destination / FACE_POSITIONS_FILENAME

        recording = self.recording
        recording_id = self._get_recording_id(recording)

        face_pos_path = destination / FACE_POSITIONS_FILENAME

        rows_written = 0
        with open(face_pos_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=None)
            for ts in sorted(face_positions.keys()):
                for face in face_positions[ts]:
                    if rows_written == 0:
                        writer.fieldnames = face.keys()
                        writer.writeheader()
                    writer.writerow(face)
                    rows_written += 1

        logger.info(f"Exported {FACE_POSITIONS_FILENAME} to {destination}")

    def _find_all_gaze_on_face(self, face_positions, gaze_data) -> dict[int, list[dict]]:
        """Helper to find gaze on face for all gaze samples, returns dict keyed by gaze timestamp."""
        recording_id = self._get_recording_id(self.recording)
        sorted_face_ts = sorted(face_positions.keys())
        gaze_on_face = {}
        for g_ts_ns, g_x, g_y in gaze_data:
            frame_ts = self._nearest_frame_ts(g_ts_ns, sorted_face_ts)
            faces_at_frame = face_positions.get(frame_ts, []) if frame_ts is not None else []
            on_face = _gaze_on_faces(g_x, g_y, faces_at_frame)
            gaze_on_face.setdefault(int(g_ts_ns), []).append({
                "recording_id": recording_id,
                "g_ts_ns": g_ts_ns,
                "g_x": round(g_x, 2),
                "g_y": round(g_y, 2),
                "gaze_on_face": on_face,
            })
        return gaze_on_face

    def _export_gaze_on_face(self, gaze_on_face: dict, destination: pathlib.Path = pathlib.Path()) -> None:
        """Background generator: writes gaze_on_face.csv using face position results and gaze data."""
        destination.mkdir(parents=True, exist_ok=True)
        gaze_pos_path = destination / GAZE_ON_FACE_FILENAME

        recording = self.recording
        recording_id = self._get_recording_id(recording)
        sorted_gaze_ts = sorted(gaze_on_face.keys())

        with open(gaze_pos_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=None)
            rows_written = 0
            for ts in sorted_gaze_ts:
                for gaze in gaze_on_face[ts]:
                    if rows_written == 0:
                        writer.fieldnames = gaze.keys()
                        writer.writeheader()
                    writer.writerow(gaze)
                    rows_written += 1
        logger.info(f"Exported {GAZE_ON_FACE_FILENAME} to {destination}")

    def _find_all_fixations_on_face(self, face_positions, fixation_data) -> dict[int, list[dict]]:
        """Helper to find gaze on face for all fixations, returns dict keyed by fixation id."""
        recording_id = self._get_recording_id(self.recording)
        sorted_face_ts = sorted(face_positions.keys())
        fixation_on_face = {}
        for fix_id, start_ns, end_ns, cx, cy in fixation_data:
            mid_ns = (start_ns + end_ns) // 2
            frame_ts = self._nearest_frame_ts(mid_ns, sorted_face_ts)
            faces_at_frame = face_positions.get(frame_ts, []) if frame_ts is not None else []
            on_face = _gaze_on_faces(cx, cy, faces_at_frame)
            fixation_on_face.setdefault(int(fix_id), []).append({
                "recording_id": recording_id,
                "fix_id": fix_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "centroid_x": round(cx, 2),
                "centroid_y": round(cy, 2),
                "fixation_on_face": on_face,
            })
        return fixation_on_face
    
    def _export_fixations_on_face(self, fixations_on_face: dict, destination: pathlib.Path = pathlib.Path()) -> None:
        """Background generator: writes fixations_on_face.csv using self._results and fixation data."""
        destination.mkdir(parents=True, exist_ok=True)
        fix_path = destination / FIXATIONS_ON_FACE_FILENAME

        sorted_keys = sorted(fixations_on_face.keys())

        rows_written = 0
        with open(fix_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=None)
            for key in sorted_keys:
                rows = fixations_on_face[key]
                for row in rows:
                    if rows_written == 0:
                        writer.fieldnames = row.keys()
                        writer.writeheader()
                    writer.writerow(row)
                    rows_written += 1

        logger.info(f"Exported {FIXATIONS_ON_FACE_FILENAME} to {destination}")


    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self._load_all_from_cache()



    # def on_recording_loaded(self, recording: NeonRecording) -> None:
    #     if len(recording.blinks) == 0:
    #         return

    #     self.get_timeline().add_timeline_broken_bar(
    #         "Blinks", self.recording.blinks[["start_time", "stop_time"]]
    #     )

    # def on_disabled(self) -> None:
    #     self.get_timeline().remove_timeline_plot("Blinks")




    def _load_all_from_cache(self) -> None:
        """Load detection results from cache and update status.
        Face information is stored in self._data['face_positions'] as a dict:
        {timestamp_ns: [face1_data, face2_data, ...], ...}
        where each face_data is a dict with keys:
        'p1_x', 'p1_y', 'p2_x', 'p2_y', 'confidence', and optional landmark keys.
        """
        cache_dir = self._cache_dir()
        face_pos_path = cache_dir / FACE_POSITIONS_FILENAME
        gaze_on_face_path = cache_dir / GAZE_ON_FACE_FILENAME
        fixations_on_face_path = cache_dir / FIXATIONS_ON_FACE_FILENAME

        if not face_pos_path.exists() or not gaze_on_face_path.exists() or not fixations_on_face_path.exists():
            logger.error(f"Missing results in cache at {cache_dir}. Run detection first.")
            self._status = "Missing face position data. Run detection first."
            return


        # face positions first
        face_positions = {}
        with open(face_pos_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    face_positions.setdefault(int(row["ts_ns"]), []).append(row)
                except Exception as exc:
                    logger.warning(f"Skipping malformed row: {row} ({exc})")

        self._data['face_positions'] = face_positions
        logger.info(f"Loaded {len(face_positions)} items from face_positions data.")

        # gaze on face next
        gaze_on_face = {}
        with open(gaze_on_face_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    gaze_on_face.setdefault(int(row["g_ts_ns"]), []).append(row)
                except Exception as exc:
                    logger.warning(f"Skipping malformed gaze row: {row} ({exc})")
        self._data['gaze_on_face'] = gaze_on_face
        logger.info(f"Loaded {len(gaze_on_face)} items from gaze_on_face data.")

        # load fixations on face last (optional, may not exist)
        fixation_on_face = {}
        with open(fixations_on_face_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    fixation_on_face.setdefault(int(row["fix_id"]), []).append(row)
                except Exception as exc:
                    logger.warning(f"Skipping malformed fixation row: {row} ({exc})")
        self._data['fixations_on_face'] = fixation_on_face
        logger.info(f"Loaded {len(fixation_on_face)} items from fixations_on_face data.")

        self._data['ts_ns_list'] = self._load_world_timestamps(self.recording)

        self._on_detection_finished()

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: pathlib.Path = pathlib.Path()) -> T.Generator[ProgressUpdate, None, None]:
        """Background generator: runs all export steps sequentially."""
        logger.info("export face positions…")
        self._export_face_positions(self._data.get('face_positions', {}), destination)
        yield ProgressUpdate(.33)
        logger.info("export gaze on face…")
        self._export_gaze_on_face(self._data.get('gaze_on_face', {}), destination)
        yield ProgressUpdate(.66)
        logger.info("export fixations on face…")
        self._export_fixations_on_face(self._data.get('fixations_on_face', {}), destination)

    def render(self, painter: QPainter, time_in_recording: int) -> None:

        if self._data is None:
            logger.info("render called but no data loaded")
            return

        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        scene_ts = self._nearest_frame_ts(time_in_recording, self._data.get('ts_ns_list', []))

        face_positions = self._data.get('face_positions', {})
        faces = face_positions.get(scene_ts, [])   
        if self._draw_overlay and faces:
            for face in faces:
                try:
                    x1, y1 = int(face["p1_x"]), int(face["p1_y"])
                    x2, y2 = int(face["p2_x"]), int(face["p2_y"])
                    confidence = float(face.get("confidence", 0))
                    color = QColor(0, 255, 0) if confidence >= self._confidence_threshold else QColor(255, 0, 0)
                    painter.setPen(color)
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                    # Draw face landmarks. They seem to be always available, but check just in case.
                    keypairs = [('el_x', 'el_y'), ('er_x', 'er_y'), ('n_x', 'n_y'), ('ml_x', 'ml_y'), ('mr_x', 'mr_y')]
                    for (xkey, ykey) in keypairs:
                        if xkey not in face or ykey not in face:
                            logger.warning(f"Missing landmark keys {xkey} or {ykey} in face data: {face}")
                            continue
                        lx, ly = int(face[xkey]), int(face[ykey])
                        painter.drawEllipse(lx - 3, ly - 3, 6, 6)
                except Exception as exc:
                    logger.warning(f"Error drawing face overlay: {exc}")


    # --------------------------------------------------------- private helpers

    def _cache_dir(self) -> pathlib.Path:
        recording = self.recording
        rec_path = pathlib.Path(recording._rec_dir)
        return rec_path / ".neon_player" / "cache" / PLUGIN_NAME

    def _on_detection_finished(self) -> None:
        face_positions = self._data.get('face_positions', {})
        n_frames = len(face_positions)
        n_faces = sum(len(v) for v in face_positions.values())
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
