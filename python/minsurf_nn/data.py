"""Manifest-backed, bounded-memory NPZ sampling in normalized coordinates."""
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = next((record[k] for k in ("path", "file", "npz") if k in record), None)
            if not isinstance(name, str):
                raise ValueError(f"{path}:{line_number}: expected NPZ path in 'path'")
            record = dict(record)
            record["path"] = str((path.parent / name).resolve())
            if not Path(record["path"]).is_file():
                raise FileNotFoundError(record["path"])
            records.append(record)
    if not records:
        raise ValueError("Manifest is empty")
    return records


def split_records(records, validation_fraction=0.2, seed=42):
    """Keep explicit holdouts; otherwise split groups, never individual queries."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if any("split" in r for r in records):
        if any(r.get("split") not in ("train", "val", "validation", "test") for r in records):
            raise ValueError("Every record must have a valid split when explicit splits are used")
        train = [r for r in records if r["split"] == "train"]
        val = [r for r in records if r["split"] in ("val", "validation")]
        train_groups = {r["group_id"] for r in train if "group_id" in r}
        if any(r.get("group_id") in train_groups for r in records if r["split"] != "train"):
            raise ValueError("group_id leaks between train and holdout splits")
    else:
        groups = {}
        for record in records:
            group = str(record.get("group_id", record["path"]))
            groups.setdefault(group, []).append(record)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError("At least two independent samples/groups are required for train/val")
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
        val = [r for k in keys[:count] for r in groups[k]]
        train = [r for k in keys[count:] for r in groups[k]]
    if not train or not val:
        raise ValueError("Both train and validation splits must be nonempty")
    return train, val


def resample_boundary(points, count):
    """Uniform arclength sampling of an ordered, closed polygonal contour."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or count < 3:
        raise ValueError("Boundary must be [N>=3,3], and requested count >= 3")
    if not np.isfinite(points).all():
        raise ValueError("Boundary contains non-finite values")
    delta = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(delta, axis=1)
    keep = lengths > 1e-12
    starts, delta, lengths = points[keep], delta[keep], lengths[keep]
    if len(lengths) < 3:
        raise ValueError("Boundary has fewer than three nonzero edges")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    positions = np.arange(count) * cumulative[-1] / count
    indices = np.minimum(np.searchsorted(cumulative, positions, side="right") - 1, len(lengths) - 1)
    fraction = (positions - cumulative[indices]) / lengths[indices]
    return (starts[indices] + delta[indices] * fraction[:, None]).astype(np.float32)


class SurfaceDataset(Dataset):
    def __init__(self, records, boundary_count=128, query_count=1024, seed=42,
                 return_area=False):
        if boundary_count < 3 or query_count < 1:
            raise ValueError("boundary_count >= 3 and query_count >= 1 required")
        self.records = records
        self.boundary_count = boundary_count
        self.query_count = query_count
        self.seed = seed
        self.epoch = 0
        self.return_area = return_area

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["path"], allow_pickle=False) as sample:
            boundary = resample_boundary(sample["boundary_points"], self.boundary_count)
            query = np.asarray(sample["query_points"], dtype=np.float32)
            target = np.asarray(sample["signed_distance"], dtype=np.float32)
            if query.ndim != 2 or query.shape[1] != 3 or target.shape != (len(query),) or not len(query):
                raise ValueError(f"Invalid query/SDF shapes in {record['path']}")
            if not np.isfinite(query).all() or not np.isfinite(target).all():
                raise ValueError(f"Non-finite query/SDF in {record['path']}")
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            chosen = rng.choice(len(query), self.query_count, replace=len(query) < self.query_count)
            result = {"boundary": torch.from_numpy(boundary),
                      "query": torch.from_numpy(query[chosen].copy()),
                      "sdf": torch.from_numpy(target[chosen].copy())}
            if self.return_area:
                if "surface_vertices" not in sample or "faces" not in sample:
                    raise ValueError(f"Target mesh is required for area in {record['path']}")
                vertices = np.asarray(sample["surface_vertices"], dtype=np.float32)
                faces = np.asarray(sample["faces"], dtype=np.int64)
                triangles = vertices[faces]
                area = 0.5 * np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0],
                             triangles[:, 2] - triangles[:, 0]), axis=1).sum()
                if not np.isfinite(area) or area <= 0:
                    raise ValueError(f"Invalid target area in {record['path']}")
                result["area"] = torch.tensor(area, dtype=torch.float32)
            return result


class AugmentedChebyshevDataset(Dataset):
    """Expose compact SO(2) packs as individual contour/coefficient pairs.

    The manifest has one record per independently generated contour, while a
    pack stores several rotations.  Keeping the split at record/group level
    prevents rotated copies from leaking between training and validation.
    """

    def __init__(self, records, boundary_count=None):
        if boundary_count is not None and boundary_count < 3:
            raise ValueError("boundary_count must be at least three")
        if not records:
            raise ValueError("At least one augmentation-pack record is required")
        self.records = records
        self.boundary_count = boundary_count
        counts = []
        for record in records:
            copies = int(record.get("copies", record.get("augmented_samples", 0)))
            if copies < 1:
                raise ValueError("Every augmentation-pack record needs copies >= 1")
            counts.append(copies)
        self.offsets = np.concatenate((np.asarray([0], dtype=np.int64),
                                       np.cumsum(counts, dtype=np.int64)))

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect_right(self.offsets, index) - 1
        copy_index = int(index - self.offsets[record_index])
        record = self.records[record_index]
        with np.load(record["path"], allow_pickle=False) as pack:
            boundary = np.asarray(pack["boundary_points"][copy_index], dtype=np.float32)
            coefficients = np.asarray(pack["coefficients"][copy_index], dtype=np.float32)
            angle = float(pack["angles"][copy_index])
            degree = int(pack["degree"])
            source_index = int(pack["source_index"])
        if self.boundary_count is not None and len(boundary) != self.boundary_count:
            boundary = resample_boundary(boundary, self.boundary_count)
        if (boundary.ndim != 2 or boundary.shape[1] != 3 or
                coefficients.ndim != 1 or not np.all(np.isfinite(coefficients))):
            raise ValueError(f"Invalid rotation pack arrays in {record['path']}")
        return {
            "boundary": torch.from_numpy(boundary.copy()),
            "coefficients": torch.from_numpy(coefficients.copy()),
            "angle": torch.tensor(angle, dtype=torch.float32),
            "degree": torch.tensor(degree, dtype=torch.int64),
            "source_index": torch.tensor(source_index, dtype=torch.int64),
            "copy_index": torch.tensor(copy_index, dtype=torch.int64),
        }


class ChebyshevRotationPackDataset(Dataset):
    """Load every rotated copy in a compact pack with one NPZ open.

    Training batches have shape ``[packs, copies, points, 3]`` and can be
    flattened to independent examples after loading.  This avoids opening the
    same compressed file once for every rotation during each epoch.
    """

    def __init__(self, records, boundary_count=64):
        if boundary_count is not None and boundary_count < 3:
            raise ValueError("boundary_count must be at least three")
        if not records:
            raise ValueError("At least one augmentation-pack record is required")
        self.records = records
        self.boundary_count = boundary_count

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["path"], allow_pickle=False) as pack:
            boundaries = np.asarray(pack["boundary_points"], dtype=np.float32)
            coefficients = np.asarray(pack["coefficients"], dtype=np.float32)
            angles = np.asarray(pack["angles"], dtype=np.float32)
            degree = int(pack["degree"])
            source_index = int(pack["source_index"])
        if (boundaries.ndim != 3 or boundaries.shape[2] != 3 or
                coefficients.ndim != 2 or len(boundaries) != len(coefficients) or
                angles.shape != (len(boundaries),) or
                not np.isfinite(boundaries).all() or
                not np.isfinite(coefficients).all()):
            raise ValueError(f"Invalid rotation pack arrays in {record['path']}")
        if self.boundary_count is not None and boundaries.shape[1] != self.boundary_count:
            boundaries = np.stack([
                resample_boundary(boundary, self.boundary_count)
                for boundary in boundaries
            ])
        return {
            "boundary": torch.from_numpy(boundaries.copy()),
            "coefficients": torch.from_numpy(coefficients.copy()),
            "angle": torch.from_numpy(angles.copy()),
            "degree": torch.tensor(degree, dtype=torch.int64),
            "source_index": torch.tensor(source_index, dtype=torch.int64),
        }
