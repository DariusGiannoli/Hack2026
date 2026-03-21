"""
camera.py — ZED camera wrapper using the ZED SDK natively.

Uses sl.Camera for frame capture and sl.BodyTracking (BODY_38) to detect the
human skeleton.  get_frames() returns (left_bgr, right_bgr) exactly as before
so the rest of the pipeline is unchanged.  get_bodies() exposes the raw ZED
Bodies object for the detector classes.

Falls back to OpenCV SBS-split if the ZED SDK is not available or the camera
fails to open.
"""

import cv2
import numpy as np

try:
    import pyzed.sl as sl
    _ZED_SDK_OK = True
except ImportError:
    _ZED_SDK_OK = False


class ZEDCamera:
    """
    ZED 2i camera wrapper.

    Parameters
    ----------
    camera_id : int
        Ignored when using the ZED SDK (the SDK finds the camera automatically).
        Used as OpenCV device index in fallback mode.
    y_offset : int
        Vertical pixel shift applied to the right frame to correct lens
        misalignment.  Positive → shifts right frame DOWN.
    enable_body_tracking : bool
        Enable ZED body tracking (BODY_38).  Set False to save GPU if you only
        need raw frames.
    """

    def __init__(self, camera_id: int = 0, y_offset: int = 0,
                 enable_body_tracking: bool = True):
        self.y_offset = y_offset
        self._bodies = None
        self._use_sdk = _ZED_SDK_OK

        if self._use_sdk:
            self._init_sdk(enable_body_tracking)
        else:
            print("[ZEDCamera] pyzed not available — falling back to OpenCV.")
            self._init_opencv(camera_id)

    # ── ZED SDK path ────────────────────────────────────────────────────────

    def _init_sdk(self, enable_body_tracking: bool):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60   # HD720 supporte 30 ou 60 fps sur ZED 2i
        init_params.coordinate_units = sl.UNIT.METER
        # RIGHT_HANDED_Y_UP: X=right, Y=up, Z=toward camera (same as MediaPipe world)
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        self._zed = sl.Camera()
        err = self._zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZEDCamera] ZED SDK open failed ({err}) — falling back to OpenCV.")
            self._use_sdk = False
            self._init_opencv(0)
            return

        # Pull real calibration and update the geometry singleton
        cal = self._zed.get_camera_information().camera_configuration.calibration_parameters
        lc = cal.left_cam
        from vision.geometry import ZED2I
        ZED2I.fx = float(lc.fx)
        ZED2I.fy = float(lc.fy)
        ZED2I.cx = float(lc.cx)
        ZED2I.cy = float(lc.cy)
        ZED2I.baseline_m = float(cal.get_camera_baseline())
        print(
            f"[ZEDCamera] Calibration ZED SDK: "
            f"fx={ZED2I.fx:.1f} fy={ZED2I.fy:.1f} "
            f"cx={ZED2I.cx:.1f} cy={ZED2I.cy:.1f} "
            f"B={ZED2I.baseline_m:.4f} m"
        )

        self._image_l = sl.Mat()
        self._image_r = sl.Mat()
        self._point_cloud = sl.Mat()   # XYZRGBA, used for hand depth lifting
        self._body_tracking_enabled = False

        if enable_body_tracking:
            # Positional tracking must be enabled before body tracking
            pt_params = sl.PositionalTrackingParameters()
            pt_params.enable_imu_fusion = True
            err = self._zed.enable_positional_tracking(pt_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[ZEDCamera] Positional tracking non disponible ({err}).")

            body_params = sl.BodyTrackingParameters()
            body_params.enable_tracking = True
            body_params.enable_body_fitting = True
            body_params.body_format = sl.BODY_FORMAT.BODY_38
            body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
            body_params.body_selection = sl.BODY_KEYPOINTS_SELECTION.FULL
            err = self._zed.enable_body_tracking(body_params)
            if err == sl.ERROR_CODE.SUCCESS:
                self._body_runtime = sl.BodyTrackingRuntimeParameters()
                self._body_runtime.detection_confidence_threshold = 40
                self._bodies_obj = sl.Bodies()
                self._body_tracking_enabled = True
                print("[ZEDCamera] Body tracking BODY_38 activé.")
            else:
                print(f"[ZEDCamera] Body tracking non disponible ({err}).")

    def _get_frames_sdk(self):
        if self._zed.grab() != sl.ERROR_CODE.SUCCESS:
            return None, None

        self._zed.retrieve_image(self._image_l, sl.VIEW.LEFT)
        self._zed.retrieve_image(self._image_r, sl.VIEW.RIGHT)

        # ZED SDK returns BGRA — drop alpha channel
        frame_l = self._image_l.get_data()[:, :, :3].copy()
        frame_r = self._image_r.get_data()[:, :, :3].copy()

        # Point cloud aligned on left image (used by StereoHandTracker for depth)
        self._zed.retrieve_measure(self._point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

        if self._body_tracking_enabled:
            self._zed.retrieve_bodies(self._bodies_obj, self._body_runtime)
            self._bodies = self._bodies_obj

        if self.y_offset != 0:
            M = np.float32([[1, 0, 0], [0, 1, self.y_offset]])
            frame_r = cv2.warpAffine(frame_r, M, (frame_r.shape[1], frame_r.shape[0]))

        return frame_l, frame_r

    # ── OpenCV fallback path ────────────────────────────────────────────────

    def _init_opencv(self, camera_id: int):
        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)  # backend V4L2 = latence moindre
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG = moins de bande passante USB
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        actual_w   = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h   = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[ZEDCamera] OpenCV fallback: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f} fps")
        if not self._cap.isOpened():
            print(f"[ZEDCamera] Warning: camera {camera_id} could not be opened.")

    def _get_frames_opencv(self):
        success, frame = self._cap.read()
        if not success:
            return None, None
        h, w, _ = frame.shape
        half_w = w // 2
        left = frame[:, :half_w]
        right = frame[:, half_w:]
        if self.y_offset != 0:
            M = np.float32([[1, 0, 0], [0, 1, self.y_offset]])
            right = cv2.warpAffine(right, M, (right.shape[1], right.shape[0]))
        return left, right

    # ── Public interface ────────────────────────────────────────────────────

    def get_frames(self):
        """Return (frame_left_bgr, frame_right_bgr).  Triggers body tracking."""
        if self._use_sdk:
            return self._get_frames_sdk()
        return self._get_frames_opencv()

    def get_bodies(self):
        """Return the latest ZED Bodies object, or None in OpenCV fallback mode."""
        return self._bodies

    def get_point_cloud(self):
        """Return the latest ZED point cloud sl.Mat (XYZRGBA, CPU), or None."""
        if self._use_sdk:
            return self._point_cloud
        return None

    @property
    def using_zed_sdk(self) -> bool:
        return self._use_sdk

    def close(self):
        if self._use_sdk:
            self._zed.close()
        else:
            self._cap.release()
