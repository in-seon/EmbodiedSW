import time
from dataclasses import dataclass, field

from config import config
from src.capture import CameraCapture
from src.detection import PersonDetector
from src.fall_detection import FallDetectionPipeline, roi_from_ratio, roi_from_zones
from src.serial_comm import SerialComm
from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE
from src.signal_extend import Occupant, maximum_eta_sec, minimum_progress
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrossingProgress, CrosswalkZones


@dataclass
class PedestrianState:
    track_id: object
    foot_point: tuple
    zone: object = None             
    progress: object = None          
    direction: object = None         
    confirmed: bool = False         
    speed: object = None             
    crossing_time_sec: object = None 

@dataclass
class FrameResult:
    progress: object = None
    eta_sec: object = None
    occupied_zones: list = field(default_factory=list)
    pedestrians: list = field(default_factory=list)  
    speed_unit: str = "px/s"       
    untracked_count: int = 0

    def serial_state(self):
        if self.progress is not None:
            return STATE_ZONE, self.progress
        return STATE_NORMAL, None


class SignalExtensionPipeline:
    def __init__(self, camera=None, detector=None, zones=None, occupancy=None,
                 serial_comm=None, speed_estimator=None, progress=None):

        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        self.zones = zones or CrosswalkZones.load()
        self.occupancy = occupancy or CrosswalkOccupancy(self.zones)
        self.progress = progress or CrossingProgress(zone_count=len(self.zones))
        self.serial_comm = serial_comm or SerialComm()
        self.speed = speed_estimator or SpeedEstimator(ground_plane=self.zones.ground_plane)

    def process_frame(self, frame, timestamp=None) -> FrameResult:
        boxes = self.detector.detect(frame)
        return self.process_boxes(boxes, timestamp)

    def process_boxes(self, boxes, timestamp=None) -> FrameResult:
        if timestamp is None:
            timestamp = time.monotonic()

        detections = [
            (box.track_id, box.foot_point())
            for box in boxes
            if box.is_pedestrian()
        ]
        confirmed = self.occupancy.update(detections)
        for old_id, new_id in self.occupancy.rekeyed:
            self.progress.rekey(old_id, new_id)

        occupied = self.occupancy.occupied_zones()
        speeds = self.speed.update_many(detections, timestamp)

        progresses = {}
        for track_id, zone in confirmed.items():
            progresses[track_id] = self.progress.update(track_id, zone)
        self.progress.keep_only(confirmed.keys())

        pedestrians = [
            PedestrianState(
                track_id=track_id,
                foot_point=foot_point,
                zone=self.zones.locate(foot_point),
                progress=progresses.get(track_id),
                direction=self.progress.direction(track_id),
                confirmed=track_id in confirmed,
                speed=speeds.get(track_id),
                crossing_time_sec=self.speed.estimated_crossing_time_sec(track_id),
            )
            for track_id, foot_point in detections
        ]

        occupants = [
            Occupant(progress=value,
                     eta_sec=self.speed.estimated_crossing_time_sec(track_id))
            for track_id, value in progresses.items()
        ]
        progress_zone = minimum_progress(occupants)
        eta_sec = maximum_eta_sec(occupants)    

        return FrameResult(
            progress=progress_zone,
            eta_sec=eta_sec,
            occupied_zones=occupied,
            pedestrians=pedestrians,
            speed_unit=self.speed.unit,
            untracked_count=self.occupancy.untracked_count,
        )

    def run(self, on_result=None):
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                result = self.process_frame(frame)
                state, zone = result.serial_state()
                self.serial_comm.update_state(state, zone)
                if on_result is not None and on_result(result, frame) is False:
                    break



@dataclass
class FallAlarmResult:
    fall_confirmed: bool = False
    confirmed_ids: set = field(default_factory=set)
    people_count: int = 0
    command_sent: object = None


class FallAlarmPipeline:
    def __init__(self, camera=None, detector=None, serial_comm=None,
                 zones=None, roi_px=None, fall_detector=None):
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        self.serial_comm = serial_comm or SerialComm()
        self.zones = zones
        self.roi_px = roi_px
        self._fall = fall_detector

    def _fall_pipeline(self, frame):
        if self._fall is None:
            roi = self.roi_px
            if roi is None:
                roi = (
                    roi_from_zones(self.zones) if self.zones is not None
                    else roi_from_ratio(frame.shape)
                )
                self.roi_px = roi
            self._fall = FallDetectionPipeline(roi)
        return self._fall

    def process_frame(self, frame, now=None) -> FallAlarmResult:
        now = time.monotonic() if now is None else now
        boxes = self.detector.detect(frame)
        return self.process_boxes(boxes, frame, now)

    def process_boxes(self, boxes, frame, now=None) -> FallAlarmResult:
        now = time.monotonic() if now is None else now

        fall = self._fall_pipeline(frame).update(boxes, now)
        state = STATE_FALL if fall["fall_confirmed"] else STATE_NORMAL
        command = self.serial_comm.update_state(state, now=now)

        return FallAlarmResult(
            fall_confirmed=fall["fall_confirmed"],
            confirmed_ids=fall["confirmed_ids"],
            people_count=len(fall["people"]),
            command_sent=command,
        )

    def reset_alarm(self):
        if self._fall is not None:
            self._fall.reset()
        self.serial_comm.send_state(STATE_NORMAL)

    def run(self, on_result=None):
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                result = self.process_frame(frame)
                if on_result is not None and on_result(result, frame) is False:
                    break


@dataclass
class CombinedResult:
    fall: FallAlarmResult = field(default_factory=FallAlarmResult)
    extension: FrameResult = field(default_factory=FrameResult)
    state: str = STATE_NORMAL         
    zone: object = None                
    line_sent: object = None          


class CombinedPipeline:
    def __init__(self, camera=None, detector=None, serial_comm=None,
                 extension=None, fall=None, zones=None):
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        self.serial_comm = serial_comm or SerialComm()
        zones = zones if zones is not None else CrosswalkZones.load()

        self.extension = extension or SignalExtensionPipeline(
            camera=self.camera, detector=self.detector,
            serial_comm=_NoSend(), zones=zones,
        )
        self.fall = fall or FallAlarmPipeline(
            camera=self.camera, detector=self.detector,
            serial_comm=_NoSend(), zones=zones,
        )

    def process_frame(self, frame, timestamp=None) -> CombinedResult:

        now = time.monotonic() if timestamp is None else timestamp
        boxes = self.detector.detect(frame)
        fall_result = self.fall.process_boxes(boxes, frame, now)
        ext_result = self.extension.process_boxes(boxes, now)

        if fall_result.fall_confirmed:
            state, zone = STATE_FALL, None
        else:
            state, zone = ext_result.serial_state()

        line = self.serial_comm.update_state(state, zone, now=now)
        return CombinedResult(
            fall=fall_result, extension=ext_result,
            state=state, zone=zone, line_sent=line,
        )

    def reset_alarm(self):
        self.fall.reset_alarm()

    def run(self, on_result=None):
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                result = self.process_frame(frame)
                if on_result is not None and on_result(result, frame) is False:
                    break


class _NoSend:
    
    def update_state(self, state, extend_sec=None, now=None):
        return None

    def send_state(self, state, extend_sec=None, now=None):
        return None

