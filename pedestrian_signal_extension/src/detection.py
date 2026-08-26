"""검출 모델 래퍼 (COCO 사전학습 yolov8n).

CLAUDE.md 2.6: 실측 확인 결과 COCO 사전학습 yolov8n이 사람 모형을 "person"으로 검출했고
신뢰도는 80%대였다. 따라서 추가 파인튜닝 없이 사전학습 가중치를 그대로 쓴다.

교통약자(휠체어/목발/지팡이)는 **별도 가중치를 쓰는 MobilityAidDetector**가 맡는다(CLAUDE.md 2.5).
사람 검출 가중치에는 해당 클래스가 없으므로 PersonDetector로는 원리상 잡을 수 없다.
config.MOBILITY_AID_MODEL_PATH가 None이면 그 검출기는 조용히 비활성되어 빈 목록을 주고,
결과적으로 SignalExtensionPipeline의 priority_mode가 항상 False가 된다.

track_id는 YOLO.track(persist=True)가 부여한다. speed.py와 zone.py의 CrosswalkOccupancy가
사람을 프레임 간에 구분하려면 이 ID가 필요하다.
"""

from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str                     # config.PEDESTRIAN_LABEL 또는 config.MOBILITY_AID_LABELS 중 하나(모델 클래스명과 일치).
    track_id: Optional[int] = None  # 추적 ID. YOLO track()이 부여. 없으면 None.
    # COCO 17개 관절의 (x, y, conf) 배열. 포즈 가중치(yolov8n-pose.pt)일 때만 채워진다.
    # 목표 2(쓰러짐 감지)가 몸통 각도를 재는 데 쓴다 — src/fall_detection.py 참고.
    # 일반 검출 가중치면 None이고, 그때 쓰러짐 판정은 bbox 가로/세로 비율로 폴백한다.
    keypoints: object = None

    def foot_point(self):
        """발 위치 = bounding box 하단 모서리의 중심. 위치 판정·속도 추정의 기준점.

        카메라가 횡단보도를 사선으로 비추므로 사람의 전신이 대체로 보이고, 박스 아래쪽이
        지면에 닿는 지점에 대응한다. 반면 center_point()는 사람 키의 절반만큼 위에 떠 있어
        지면 좌표로 쓰면 실제 서 있는 위치보다 카메라 쪽으로 당겨진 지점으로 잘못 판정된다
        (CLAUDE.md 2.1). 그래서 zone 판정과 호모그래피 변환에는 항상 이 점을 쓴다.
        """
        return ((self.x1 + self.x2) / 2, self.y2)

    def center_point(self):
        """박스 중심. **지면 위치 판정에는 쓰지 말 것**(위 foot_point 주석 참고).

        지면 좌표가 필요 없는 용도로만 쓴다. 현재 실사용처는 없고,
        tests/test_pipeline.py::test_uses_foot_point_not_center_for_zone 이
        "발 위치를 쓰지 중심점을 쓰지 않는다"는 설계 결정을 이 값과 대조해 고정한다.
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_pedestrian(self) -> bool:
        return self.label == config.PEDESTRIAN_LABEL

    def is_mobility_aid(self) -> bool:
        """교통약자 보조기구 클래스인가.

        MobilityAidDetector가 낸 박스에만 의미가 있다 — PersonDetector는 사람만 보므로
        그쪽 박스에서는 항상 False다.
        """
        return self.label in config.MOBILITY_AID_LABELS


class PersonDetector:
    """ultralytics YOLO 래퍼. 프레임당 추론 1회로 **사람만** 검출한다.

    교통약자 보조기구(휠체어/목발)는 이 검출기의 일이 아니다 — 사람 검출 가중치에 해당
    클래스가 없어서 원리상 못 잡는다. 그쪽은 별도 가중치를 쓰는 MobilityAidDetector가 맡는다.

    detect()는 추적 ID가 붙은 BoundingBox 리스트를 반환한다. 모델을 파인튜닝 가중치로
    교체하더라도 이 시그니처가 유지되므로 zone/speed/signal_extend 쪽은 손댈 필요가 없다.
    """

    def __init__(self, model_path=None, confidence_threshold=None, tracker=None, imgsz=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.tracker = tracker or config.DETECTION_TRACKER
        # 추론 입력 해상도. 프레임 해상도(CAMERA_RESOLUTION)와 다른 값이며, 이걸 줄여도
        # 박스는 원본 프레임 좌표로 돌아오므로 zone/호모그래피 재캘리브레이션이 필요 없다.
        self.imgsz = imgsz if imgsz is not None else config.DETECTION_IMGSZ
        if self.model_path is None:
            raise NotImplementedError(
                "DETECTION_MODEL_PATH가 설정되지 않았습니다. config에 가중치 경로를 채워주세요."
            )
        self._model = self._load_model(self.model_path)
        # 모델이 아는 클래스 이름 {인덱스: 이름}. detect()에서 라벨 문자열로 되돌리는 데 쓴다.
        self._class_names = self._model.names

        # 관심 있는 클래스만 추론 단계에서 걸러 불필요한 박스를 만들지 않는다.
        #
        # 여기에 교통약자 라벨을 섞지 않는다. 사람 검출 가중치(yolov8n-pose)에는 휠체어·목발
        # 클래스가 없어서 넣어봐야 걸리는 것이 없고, "이 검출기도 교통약자를 본다"는 인상만
        # 남아 실제 담당자(MobilityAidDetector)와 역할이 겹쳐 보이게 된다.
        # 이 클래스는 사람만 본다.
        self._class_ids = sorted(
            idx for idx, name in self._class_names.items()
            if name == config.PEDESTRIAN_LABEL
        )
        if not self._class_ids:
            raise ValueError(
                f"모델 {self.model_path}에 '{config.PEDESTRIAN_LABEL}' 클래스가 없습니다. "
                f"모델이 아는 클래스: {sorted(self._class_names.values())}. "
                "config.PEDESTRIAN_LABEL이 모델 클래스명과 일치하는지 확인하세요."
            )

    def _load_model(self, model_path):
        # 지연 임포트: ultralytics는 무거워서, 이 클래스를 쓰지 않는 테스트/도구가
        # 임포트 비용을 물지 않게 한다.
        from ultralytics import YOLO

        return YOLO(model_path)

    def detect(self, frame) -> list[BoundingBox]:
        """frame에서 관심 클래스를 검출해 BoundingBox 리스트를 반환한다.

        track(persist=True)를 쓰므로 같은 사람에게는 프레임 간에 같은 track_id가 붙는다.
        추적기가 ID를 붙이지 못한 검출은 track_id=None으로 나가며, 속도 추정에서는 무시된다.
        """
        results = self._model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            classes=self._class_ids,
            tracker=self.tracker,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        boxes = []
        if results.boxes is None:
            return boxes

        # 포즈 가중치면 같은 추론 결과에 키포인트가 함께 들어 있다. 추론을 한 번 더 돌리지
        # 않고 그대로 실어 보내, 목표 1(신호 연장)과 목표 2(쓰러짐)가 추론을 공유하게 한다.
        keypoints = None
        if getattr(results, "keypoints", None) is not None:
            keypoints = results.keypoints.data
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()

        for index, box in enumerate(results.boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = self._class_names[int(box.cls[0])]
            track_id = int(box.id[0]) if box.id is not None else None
            kp = keypoints[index] if keypoints is not None and index < len(keypoints) else None
            boxes.append(
                BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(box.conf[0]),
                    label=label,
                    track_id=track_id,
                    keypoints=kp,
                )
            )
        return boxes

    def reset_tracker(self):
        """보행 신호 사이클이 바뀌는 등, 추적 상태를 끊고 싶을 때 호출."""
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    tracker.reset()


class MobilityAidDetector:
    """교통약자(휠체어/목발/지팡이) 보조 검출기 — 별도 가중치를 저빈도로 돌린다.

    사람 검출과 분리한 이유는 단순하다: COCO에도 yolov8n-pose에도 해당 클래스가 없어
    같은 모델로는 못 잡는다(CLAUDE.md 2.5).

    **매 프레임 돌리지 않는다.** 추론이 프레임 시간의 사실상 전부라 두 모델을 매 프레임
    돌리면 FPS가 반토막 나는데, "이 사람이 휠체어를 탔는가"는 위치·속도와 달리
    프레임마다 바뀌는 값이 아니다. every_n_frames 마다 한 번만 추론하고 그 사이에는
    직전 결과를 그대로 돌려준다(값이 깜빡이지 않게).

    가중치가 없으면(config.MOBILITY_AID_MODEL_PATH is None) 조용히 비활성 상태가 되어
    항상 빈 목록을 돌려준다 — 보류 중인 기능이 나머지를 막지 않게 하기 위함이다.
    """

    def __init__(self, model_path=None, confidence_threshold=None, imgsz=None,
                 every_n_frames=None, labels=None):
        self.model_path = model_path or config.MOBILITY_AID_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.MOBILITY_AID_CONFIDENCE_THRESHOLD
        )
        self.imgsz = imgsz if imgsz is not None else config.MOBILITY_AID_IMGSZ
        self.every_n_frames = (
            every_n_frames if every_n_frames is not None
            else config.MOBILITY_AID_EVERY_N_FRAMES
        )
        # detect()가 이 값으로 나머지 연산을 하므로 0이면 ZeroDivisionError가 난다.
        # 추론 주기를 0으로 두는 건 의미도 없으니 생성 시점에 막는다.
        if self.every_n_frames is None or self.every_n_frames < 1:
            raise ValueError(
                f"every_n_frames는 1 이상이어야 합니다(1 = 매 프레임): {self.every_n_frames}"
            )
        self._wanted = tuple(labels) if labels is not None else tuple(config.MOBILITY_AID_LABELS)
        self._frame_count = 0
        self._cached = []
        self.inference_count = 0

        self._model = None
        self._class_names = {}
        self._class_ids = None
        if self.model_path is not None:
            self._model = self._load_model(self.model_path)
            self._class_names = self._model.names
            # 라벨을 지정하지 않으면 모델이 아는 클래스를 전부 쓴다 — 후보 가중치를
            # 처음 시험할 때 클래스명을 몰라도 바로 돌려볼 수 있게.
            if self._wanted:
                self._class_ids = sorted(
                    idx for idx, name in self._class_names.items() if name in self._wanted
                )
                if not self._class_ids:
                    raise ValueError(
                        f"모델 {self.model_path}에 {list(self._wanted)} 클래스가 없습니다. "
                        f"모델이 아는 클래스: {sorted(self._class_names.values())}. "
                        "config.MOBILITY_AID_LABELS를 확인하세요(비워 두면 전체 클래스를 씁니다)."
                    )

    @property
    def enabled(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict:
        """모델이 아는 클래스 {인덱스: 이름}. 후보 가중치 확인용."""
        return dict(self._class_names)

    def _load_model(self, model_path):
        from ultralytics import YOLO   # 지연 임포트 (PersonDetector와 같은 이유)

        try:
            return YOLO(model_path)
        except Exception as exc:
            # 여기서 걸리는 원인은 대부분 하나다: 가중치 파일이 실제로는 없는 것.
            # 이름만 있고 파일이 없으면 ultralytics가 공개 허브에서 내려받으려 하는데,
            # 파인튜닝 가중치는 허브에 없으므로 "무슨 소린지 모를" 에러로 끝난다.
            raise RuntimeError(
                f"교통약자 보조 모델을 열 수 없습니다: {model_path}\n"
                f"  ({exc})\n"
                "  - 파일이 실제로 그 경로에 있는지 확인하세요. 파인튜닝 가중치는 공개 허브에\n"
                "    없으므로 자동 다운로드가 되지 않습니다(사람 검출용 yolov8n-pose.pt와 다릅니다).\n"
                "  - 아직 후보 가중치가 없다면 config.MOBILITY_AID_MODEL_PATH = None 으로 두세요.\n"
                "    보조 검출만 조용히 꺼지고 나머지는 그대로 동작합니다."
            ) from exc

    def detect(self, frame) -> list[BoundingBox]:
        """교통약자 클래스 BoundingBox 목록. 추론을 건너뛴 프레임에서는 직전 결과를 준다.

        track_id는 붙이지 않는다(track()이 아니라 predict()). 이 검출은 '있냐 없냐'만
        쓰이고, 사람과의 연결이 필요해지면 사람 박스와의 겹침으로 잇는 편이 낫다.
        """
        if self._model is None:
            return []
        if self._frame_count % self.every_n_frames != 0:
            self._frame_count += 1
            return self._cached
        self._frame_count += 1
        self.inference_count += 1

        kwargs = dict(conf=self.confidence_threshold, imgsz=self.imgsz, verbose=False)
        if self._class_ids:
            kwargs["classes"] = self._class_ids
        results = self._model.predict(frame, **kwargs)[0]

        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                boxes.append(
                    BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(box.conf[0]),
                        label=self._class_names[int(box.cls[0])],
                    )
                )
        self._cached = boxes
        return boxes
