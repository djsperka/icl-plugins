# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import csv
import logging
import os
import pathlib
import typing as T

import cv2
import numpy as np
import time

from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from qt_property_widgets.utilities import property_params
from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLUGIN_NAME = "Test plugin"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class TestPlugin(Plugin):
    """
    RetinaFace-R50 face detection for Neon Player.

    Produces the same CSV outputs as Pupil Cloud's Face Mapper Enrichment:
      • face_positions.csv     – bounding boxes & 5 facial landmarks per frame
      • gaze_on_face.csv       – whether each gaze sample falls on a face
      • fixations_on_face.csv  – whether each fixation falls on a face
    """

    # ------------------------------------------------------------------ init

    uniqueness = "unique_by_class"


    def __init__(self) -> None:
        super().__init__()
        self._results: dict[int, int] = {}  # ts_ns -> list of face dicts
        self._time = time.time()
        logger.info(f"[TestPlugin] Initialized at {self._time}")

    # ---------------------------------------------------------------- properties

    @action
    def run_detection(self) -> None:
        """Detect faces across the whole recording (runs in background)."""
        job = self.job_manager.run_background_action(
            "Test plugin",
            "TestPlugin._detect_all_frames",
        )
        job.finished.connect(self._on_detection_finished)

    def _detect_all_frames(self) -> T.Generator[ProgressUpdate, None, None]:

        recording = self.recording
        rec_path = pathlib.Path(recording._rec_dir)
        logger.info(f"[TestPlugin] _detect_all_frames started")
        logger.info(f"[TestPlugin] Recording path: {rec_path}")

        scene_video_path = self._find_scene_video(recording)
        if scene_video_path is None:
            logger.error("[TestPlugin] No scene video found.")
            return

        cap = cv2.VideoCapture(str(scene_video_path))
        if not cap.isOpened():
            logger.error(f"[TestPlugin] Cannot open {scene_video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps_ns = self._load_world_timestamps(recording)

        #self._results = {}
        frame_idx = 0

        logger.info(f"[TestPlugin] Processing {total_frames} frames…")

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            ts_ns = (
                timestamps_ns[frame_idx]
                if frame_idx < len(timestamps_ns)
                else None
            )


            if ts_ns is not None:
                self._results[ts_ns] = ts_ns

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                progress = frame_idx / max(total_frames, 1)
                yield ProgressUpdate(progress)

        cap.release()
        logger.info(
            f"[TestPlugin] Detection done. "
            f"across {len(self._results)} frames."
        )
        logger.info(f"[TestPlugin] Initialized at {self._time}")

    # --------------------------------------------------------- private helpers

    def _cache_dir(self) -> pathlib.Path:
        recording = self.recording
        rec_path = pathlib.Path(recording._rec_dir)
        return rec_path / ".neon_player" / "cache" / PLUGIN_NAME

    def _on_detection_finished(self) -> None:
        n_frames = len(self._results)
        self._status = f"Done – {n_frames} frames processed"
        logger.info(f"[TestPlugin] {self._status}")
        logger.info(f"[TestPlugin] Initialized at {self._time}")

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
                    f"[RetinaFaceFaceMapper] Loaded {len(timestamps_ns)} "
                    f"timestamps from {time_file.name}"
                )
                return timestamps_ns
            except Exception as exc:
                logger.warning(
                    f"[RetinaFaceFaceMapper] Could not read {time_file.name}: {exc}"
                )

        # Log available files to help diagnose if nothing found
        logger.error(
            "[RetinaFaceFaceMapper] No scene .time file found. "
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
            logger.warning("[RetinaFaceFaceMapper] No gaze binary files found")
            return result

        try:
            timestamps = np.frombuffer(time_file.read_bytes(), dtype="<i8")
            # gaze .raw: float32 x, y pairs
            xy = np.frombuffer(raw_file.read_bytes(), dtype="<f4").reshape(-1, 2)
            for ts, (x, y) in zip(timestamps, xy):
                result.append((int(ts), float(x), float(y)))
            logger.info(f"[RetinaFaceFaceMapper] Loaded {len(result)} gaze samples")
        except Exception as exc:
            logger.warning(f"[RetinaFaceFaceMapper] Could not load gaze: {exc}")

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
            logger.warning("[RetinaFaceFaceMapper] No fixation binary files found")
            return result

        try:
            timestamps = np.frombuffer(time_file.read_bytes(), dtype="<i8")
            raw_bytes = raw_file.read_bytes()

            # Neon fixation dtype: start_ns int64, end_ns int64, x float32, y float32
            # = 8 + 8 + 4 + 4 = 24 bytes per fixation
            dtype = np.dtype([
                ("start_ns", "<i8"),
                ("end_ns", "<i8"),
                ("x", "<f4"),
                ("y", "<f4"),
            ])
            try:
                fixations = np.frombuffer(raw_bytes, dtype=dtype)
            except ValueError:
                # If dtype doesn't fit evenly, try reading fixations.dtype file
                dtype_file = rec_path / "fixations.dtype"
                if dtype_file.exists():
                    logger.info(f"[RetinaFaceFaceMapper] fixations.dtype: {dtype_file.read_text()}")
                # Fallback: treat as flat float32 and infer structure
                arr = np.frombuffer(raw_bytes, dtype="<f4")
                # Each fixation: x, y (2 floats = 8 bytes), timestamps from .time
                xy = arr.reshape(-1, 2)
                for i, (ts, (x, y)) in enumerate(zip(timestamps, xy)):
                    result.append((i + 1, int(ts), int(ts), float(x), float(y)))
                return result

            for i, (fix, ts) in enumerate(zip(fixations, timestamps)):
                result.append((
                    i + 1,
                    int(fix["start_ns"]),
                    int(fix["end_ns"]),
                    float(fix["x"]),
                    float(fix["y"]),
                ))
            logger.info(f"[RetinaFaceFaceMapper] Loaded {len(result)} fixations")
        except Exception as exc:
            logger.warning(f"[RetinaFaceFaceMapper] Could not load fixations: {exc}")

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
