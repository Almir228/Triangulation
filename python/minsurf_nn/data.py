"""Manifest-backed, bounded-memory NPZ sampling in normalized coordinates."""
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
