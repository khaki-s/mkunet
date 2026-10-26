"""Strict spatial/action contracts shared by human and optional LLM controllers."""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

REGIONS = ("WT", "TC", "ET")
ATOMIC_REGIONS = {
    "BG": (0, 0, 0),
    "ED": (1, 0, 0),
    "NCR": (1, 1, 0),
    "ET": (1, 1, 1),
}


@dataclass(frozen=True)
class SpatialROI:
    """Half-open voxel bounds in the *array's* axis order, never assumed XYZ."""
    start: Tuple[int, int, int]
    stop: Tuple[int, int, int]

    def __post_init__(self):
        for name in ("start", "stop"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or any(type(v) is not int for v in values):
                raise ValueError(f"{name} must contain three integer voxel indices")
            object.__setattr__(self, name, values)
        if any(a < 0 or b <= a for a, b in zip(self.start, self.stop)):
            raise ValueError("ROI must be nonempty and nonnegative")

    @property
    def slices(self):
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop))

    @property
    def shape(self):
        return tuple(b - a for a, b in zip(self.start, self.stop))

    def validate(self, shape):
        if len(shape) != 3 or any(b > n for b, n in zip(self.stop, shape)):
            raise ValueError("ROI extends outside the current image grid")
        return self

    def contains(self, point):
        return all(a <= p < b for p, a, b in zip(point, self.start, self.stop))


@dataclass(frozen=True)
class PointPrompt:
    point_id: str
    voxel: Tuple[int, int, int]
    target: str
    positive: bool

    def __post_init__(self):
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be a nonempty string")
        if self.target not in REGIONS:
            raise ValueError(f"point target must be one of {REGIONS}")
        if type(self.positive) is not bool:
            raise ValueError("positive must be a boolean")
        values = tuple(self.voxel)
        if len(values) != 3 or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("voxel must contain three nonnegative integer indices")
        object.__setattr__(self, "voxel", values)


@dataclass(frozen=True)
class BoxPrompt:
    """An annotated box, distinct from the 3D region allowed to change.

    Bounds are half-open in the native array grid. A 2D slice box has one
    dimension of thickness one; the editing ROI can still be fully 3D.
    """
    box_id: str
    start: Tuple[int, int, int]
    stop: Tuple[int, int, int]
    target: str
    positive: bool

    def __post_init__(self):
        if not isinstance(self.box_id, str) or not self.box_id:
            raise ValueError("box_id must be a nonempty string")
        bounds = SpatialROI(self.start, self.stop)
        object.__setattr__(self, "start", bounds.start)
        object.__setattr__(self, "stop", bounds.stop)
        if self.target not in REGIONS or type(self.positive) is not bool:
            raise ValueError("Box target must be WT/TC/ET and polarity must be boolean")

    @property
    def bounds(self):
        return SpatialROI(self.start, self.stop)


@dataclass(frozen=True)
class StructuredAction:
    target: str
    operation: str
    roi_id: str
    positive_point_ids: Tuple[str, ...] = ()
    negative_point_ids: Tuple[str, ...] = ()
    replacement_class: Optional[str] = None
    box_prompt: bool = False
    box_id: Optional[str] = None

    def __post_init__(self):
        if self.target not in REGIONS:
            raise ValueError(f"target must be one of {REGIONS}")
        if self.operation not in ("include", "exclude", "replace"):
            raise ValueError("operation must be include, exclude, or replace")
        if not isinstance(self.roi_id, str) or not self.roi_id:
            raise ValueError("roi_id must name an existing registered ROI")
        for name in ("positive_point_ids", "negative_point_ids"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{name} must be a list of registered point IDs")
            if any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{name} contains an invalid ID")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate IDs")
            object.__setattr__(self, name, tuple(values))
        if set(self.positive_point_ids) & set(self.negative_point_ids):
            raise ValueError("The same point cannot be both positive and negative")
        if type(self.box_prompt) is not bool:
            raise ValueError("box_prompt must be a boolean")
        if self.box_id is not None and (not isinstance(self.box_id, str)
                                       or not self.box_id or not self.box_prompt):
            raise ValueError("box_id requires an explicit box_prompt and a registered ID")
        if self.operation == "replace":
            if self.replacement_class not in ATOMIC_REGIONS:
                raise ValueError("replace requires explicit BG/ED/NCR/ET replacement_class")
            if self.box_prompt:
                raise ValueError("A box is not dense ground truth; replace requires marked points")
        elif self.replacement_class is not None:
            raise ValueError("replacement_class is only valid for replace")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Action must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Unknown action fields: {sorted(extra)}")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Incomplete action: {error}") from error

    def to_dict(self):
        return asdict(self)
