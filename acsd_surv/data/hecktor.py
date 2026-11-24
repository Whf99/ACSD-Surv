"""Patient-level manifests and physical-space image/mask preprocessing.

No outcome-derived value is used as an input or to fit preprocessing.  NIfTI
affines map source indices into RAS millimetres; tensor axes are D(S), H(A), W(R).
The small closed meshes are reference discretizations, not clinical contours.
"""
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from skimage.measure import marching_cubes
from torch.utils.data import Dataset


PREPROCESS_VERSION = 4


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_path(data_root, value):
    root = Path(data_root).resolve()
    path = (root / str(value).replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest path escapes data_root: %s" % value) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_fold(data_root, split_root, fold, zero_time_policy="exclude"):
    """Validate original assignments, then apply an explicit zero-time policy.

    Returns ``({train: rows, val: rows, test: rows}, audit)``.  Exclusions are
    recorded without editing manifests or moving another patient between sets.
    ``time`` remains in the unconverted unit supplied by the manifest.
    """
    if zero_time_policy not in ("exclude", "error"):
        raise ValueError("zero_time_policy must be 'exclude' or 'error'")
    fold = int(fold)
    if fold not in range(1, 6):
        raise ValueError("fold must be 1..5")
    folder = Path(split_root) / ("fold_%d" % fold)
    result, seen, seen_images, seen_labels = {}, {}, {}, {}
    audit = {"fold": fold, "zero_time_policy": zero_time_policy,
             "excluded": [], "counts": {}, "manifest_sha256": {}}
    required = {"patient_id", "stage", "time", "recur_label", "image_path", "label_path"}
    for split in ("train", "val", "test"):
        path = folder / (split + ".csv")
        raw = _read_csv(path)
        if not raw or not required.issubset(raw[0]):
            raise ValueError("Empty or incomplete manifest: %s" % path)
        audit["manifest_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for record in raw:
            row = dict(record)
            pid = str(row["patient_id"]).strip()
            if not pid or pid in seen:
                raise ValueError("Patient ID %r duplicated across %s and %s" % (pid, seen.get(pid), split))
            seen[pid] = split
            event, time = float(row["recur_label"]), float(row["time"])
            if event not in (0.0, 1.0) or not np.isfinite(time) or time < 0:
                raise ValueError("Invalid event/time for patient %s" % pid)
            for field, paths in (("image_path", seen_images), ("label_path", seen_labels)):
                source = str(_source_path(data_root, row[field])).casefold()
                if source in paths:
                    raise ValueError("Source %s reused by patients %s and %s" % (field, paths[source], pid))
                paths[source] = pid
            if time == 0:
                if zero_time_policy == "error":
                    raise ValueError("Patient %s has time=0; resolve label or select explicit exclusion" % pid)
                audit["excluded"].append({"patient_id": pid, "split": split,
                                          "reason": "time_zero_pending_definition"})
                continue
            stage = str(row["stage"]).strip()
            if stage in ("1.0", "2.0", "3.0", "4.0"):
                stage = stage.split(".")[0]
            if stage not in ("1", "2", "3", "4"):
                stage = "unknown"
            row.update(patient_id=pid, time=time, recur_label=int(event), event=int(event), stage=stage)
            rows.append(row)
        if not rows:
            raise ValueError("No eligible patients remain in %s" % path)
        result[split] = rows
        audit["counts"][split] = {"source": len(raw), "eligible": len(rows),
                                   "events": sum(r["event"] for r in rows),
                                   "stage_unknown": sum(r["stage"] == "unknown" for r in rows)}
    audit["patient_overlap"] = 0
    audit["time_unit"] = "unconverted_manifest_unit"
    return result, audit


def stage_one_hot(stage):
    vector = np.zeros(5, dtype=np.float32)
    vector[(int(stage) - 1) if str(stage) in ("1", "2", "3", "4") else 4] = 1
    return vector


def _largest_component(mask):
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError("Empty tumor mask")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = components == sizes.argmax()
    return largest, int(count), float(largest.sum() / mask.sum())


def _closed_outward(vertices, faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("Coarse mesh is not closed: each edge must occur exactly twice")
    triangle = vertices[faces] - vertices.mean(axis=0)
    signed_volume = np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])).sum() / 6
    if abs(signed_volume) < 1e-8:
        raise ValueError("Coarse mesh has negligible signed volume")
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
    return np.ascontiguousarray(faces, dtype=np.int64)


def coarse_mesh(mask, affine, max_vertices=192):
    """Bound mesh size by resampling the whole ROI, never by dropping faces.

    This intentionally coarse reference mesh retains the largest component and
    fills enclosed voxel cavities.  Every accepted mesh is closed and oriented.
    The chosen grid and component fraction are included in case metadata.
    """
    max_vertices = int(max_vertices)
    if max_vertices < 6:
        raise ValueError("mesh_max_vertices must be >=6")
    largest, count, fraction = _largest_component(mask)
    largest = ndimage.binary_fill_holes(largest)
    indices = np.argwhere(largest)
    low, high = indices.min(0), indices.max(0) + 1
    roi = largest[tuple(slice(a, b) for a, b in zip(low, high))]
    original_shape = np.array(roi.shape)
    voxel_spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    if not np.isfinite(voxel_spacing).all() or np.any(voxel_spacing <= 0):
        raise ValueError("NIfTI affine has invalid physical voxel spacing")
    physical_size = original_shape * voxel_spacing
    attempted = set()
    for longest in range(min(48, int(original_shape.max())), 0, -1):
        # Coarsen in millimetres, not raw index counts: thick-slice CT must not
        # collapse its superior/inferior dimension merely because it has few slices.
        new_shape = np.maximum(1, np.rint(physical_size * longest / physical_size.max()).astype(int))
        shape_key = tuple(new_shape)
        if shape_key in attempted:
            continue
        attempted.add(shape_key)
        step = original_shape / new_shape
        offset = (step - 1) / 2
        sampled = ndimage.affine_transform(roi.astype(np.float32), np.diag(step), offset=offset,
                                           output_shape=shape_key, order=0, mode="nearest", prefilter=False) > .5
        if not sampled.any():
            continue
        sampled, _, _ = _largest_component(sampled)
        field = ndimage.gaussian_filter(np.pad(sampled.astype(np.float32), 2), sigma=.4)
        if field.max() <= .5:
            continue
        voxels, faces, _, _ = marching_cubes(field, level=.5, allow_degenerate=False)
        if len(voxels) > max_vertices:
            continue
        original_indices = (voxels - 2) * step + offset + low
        vertices = nib.affines.apply_affine(affine, original_indices).astype(np.float32)
        faces = _closed_outward(vertices, faces.astype(np.int64))
        return vertices, faces, {"mesh_vertex_count": len(vertices), "mesh_face_count": len(faces),
                                 "mesh_grid_shape": list(map(int, new_shape)),
                                 "mesh_axis_spacing_mm": (physical_size / new_shape).tolist(),
                                 "original_component_count": count,
                                 "largest_component_fraction": fraction}
    raise ValueError("Cannot construct closed mesh within %d vertices" % max_vertices)


def _reference_table(path):
    if not path:
        return {}
    table = {}
    for row in _read_csv(path):
        pid = row["patient_id"]
        if pid in table:
            raise ValueError("Duplicate reference for patient %s" % pid)
        origin = np.array([float(row["origin_" + a]) for a in ("x", "y", "z")], dtype=np.float32)
        length = float(row["length_mm"])
        keys = ["axis_%s_%s" % (d, a) for d in ("r", "a", "s") for a in ("x", "y", "z")]
        if any(row.get(k, "") != "" for k in keys):
            if not all(row.get(k, "") != "" for k in keys):
                raise ValueError("Reference requires all nine axis values for %s" % pid)
            axes = np.array([float(row[k]) for k in keys], dtype=np.float32).reshape(3, 3).T
        else:
            axes = np.eye(3, dtype=np.float32)
        if (not np.isfinite(origin).all() or not np.isfinite(axes).all() or not np.isfinite(length)
                or length <= 0 or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(axes), 1, atol=1e-4)):
            raise ValueError("Invalid physical reference frame for %s" % pid)
        table[pid] = (origin, axes, length)
    return table


def _preprocessing_config(config):
    config = dict(config or {})
    # Aliases make configuration files explicit while preserving the CLI API.
    return {"shape": list(config.get("shape", config.get("image_shape", [48, 48, 48]))),
            "crop_mm": list(config.get("crop_mm", config.get("crop_size_mm", [160, 160, 128]))),
            "hu_clip": list(config.get("hu_clip", [-200, 300])),
            "intensity_mode": str(config.get("intensity_mode", "hu")),
            "mesh_max_vertices": int(config.get("mesh_max_vertices", 192)),
            "need_geometry": bool(config.get("need_geometry", True)),
            "reference_csv": str(config.get("reference_csv") or ""),
            "reference_length_mm": float(config.get("reference_length_mm", 100))}


class NPCDataset(Dataset):
    def __init__(self, rows, data_root, cache_dir, config=None, augment=False):
        self.rows = list(rows)
        self.data_root = Path(data_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = _preprocessing_config(config)
        if (config or {}).get("peritumor_mm"):
            self.config["peritumor_mm"] = float(config["peritumor_mm"])
        self.augment = bool(augment)
        self.references = _reference_table(self.config["reference_csv"])
        self.reference_sha256 = (hashlib.sha256(Path(self.config["reference_csv"]).read_bytes()).hexdigest()
                                 if self.config["reference_csv"] else None)
        shape, crop, hu = self.config["shape"], self.config["crop_mm"], self.config["hu_clip"]
        if len(shape) != 3 or min(shape) < 2 or len(crop) != 3 or min(crop) <= 0:
            raise ValueError("shape requires three integers >=2; crop_mm three positive lengths")
        if len(hu) != 2 or not hu[0] < hu[1]:
            raise ValueError("hu_clip must be increasing")
        if self.config["intensity_mode"] not in ("hu", "unit"):
            raise ValueError("intensity_mode must be 'hu' or 'unit'")
        if not np.isfinite(self.config["reference_length_mm"]) or self.config["reference_length_mm"] <= 0:
            raise ValueError("reference_length_mm must be finite and positive")

    def __len__(self):
        return len(self.rows)

    def _cache_path(self, row):
        sources = [_source_path(self.data_root, row[k]) for k in ("image_path", "label_path")]
        signature = {"version": PREPROCESS_VERSION, "config": self.config,
                     "reference_sha256": self.reference_sha256,
                     "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / (key + ".npz"), sources

    def _prepare(self, row, sources):
        image_nii, mask_nii = (nib.as_closest_canonical(nib.load(str(p))) for p in sources)
        if (len(image_nii.shape) != 3 or image_nii.shape != mask_nii.shape
                or not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-4, rtol=1e-6)):
            raise ValueError("Image/mask shape or affine mismatch for %s" % row["patient_id"])
        affine = image_nii.affine
        mask = np.asarray(mask_nii.dataobj) > 0
        largest, components, fraction = _largest_component(mask)
        image = image_nii.get_fdata(dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError("Non-finite CT intensities for %s" % row["patient_id"])
        native_min, native_max = float(image.min()), float(image.max())
        if self.config["intensity_mode"] == "unit":
            if native_min < -1e-4 or native_max > 1 + 1e-4:
                raise ValueError("Unit intensity mode requires source values in [0,1] for %s; got [%g,%g]"
                                 % (row["patient_id"], native_min, native_max))
            image = np.clip(image, 0, 1)
        center = nib.affines.apply_affine(affine, np.array(ndimage.center_of_mass(largest)))
        shape = np.array(self.config["shape"], dtype=int)
        crop = np.array(self.config["crop_mm"], dtype=float)
        # Columns of output-to-world map correspond to tensor D(S), H(A), W(R).
        output_to_world = np.zeros((3, 3))
        output_to_world[2, 0] = crop[2] / (shape[0] - 1)
        output_to_world[1, 1] = crop[1] / (shape[1] - 1)
        output_to_world[0, 2] = crop[0] / (shape[2] - 1)
        world_first = center - crop / 2
        inverse = np.linalg.inv(affine)
        matrix = inverse[:3, :3] @ output_to_world
        offset = inverse[:3, :3] @ world_first + inverse[:3, 3]
        lo, hi = (0., 1.) if self.config["intensity_mode"] == "unit" else self.config["hu_clip"]
        patch = ndimage.affine_transform(image, matrix, offset=offset, output_shape=tuple(shape),
                                         order=1, mode="constant", cval=float(lo), prefilter=False)
        patch = (np.clip(patch, lo, hi) - lo) / (hi - lo)
        arrays = {"image": patch[None].astype(np.float32)}
        if "peritumor_mm" in self.config:
            spacing = nib.affines.voxel_sizes(affine)
            distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
            ring = (~mask) & (distance <= self.config["peritumor_mm"])
            regions = [ndimage.affine_transform(region.astype(np.float32), matrix, offset=offset,
                       output_shape=tuple(shape), order=1, mode="constant", cval=0., prefilter=False)
                       for region in (mask, ring)]
            if any(region.sum() <= 0 for region in regions):
                raise ValueError("Empty sampled tumor or rim for " + row["patient_id"])
            arrays["image"] = np.stack([patch, *regions]).astype(np.float32)
        metadata = {"patient_id": row["patient_id"], "original_component_count": components,
                    "largest_component_fraction": fraction, "preprocess_version": PREPROCESS_VERSION,
                    "intensity_mode": self.config["intensity_mode"], "native_intensity_range": [native_min, native_max]}
        if self.config["need_geometry"]:
            vertices, faces, mesh_metadata = coarse_mesh(mask, affine, self.config["mesh_max_vertices"])
            if self.config["reference_csv"]:
                if row["patient_id"] not in self.references:
                    raise ValueError("Missing supplied anatomical reference for %s" % row["patient_id"])
                origin, axes, length = self.references[row["patient_id"]]
                mode = "external_reference"
            else:
                origin = nib.affines.apply_affine(affine, (np.array(image_nii.shape) - 1) / 2)
                axes, length = np.eye(3), self.config["reference_length_mm"]
                mode = "fov_proxy"
            arrays.update(vertices=vertices, faces=faces, reference_origin=np.asarray(origin, dtype=np.float32),
                          reference_axes=np.asarray(axes, dtype=np.float32),
                          reference_length=np.array(length, dtype=np.float32))
            metadata.update(mesh_metadata, reference_mode=mode)
        else:
            metadata.update(reference_mode="not_used", mesh_vertex_count=0)
        arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
        return arrays

    def __getitem__(self, index):
        row = self.rows[index]
        cache_path, sources = self._cache_path(row)
        if not cache_path.exists():
            arrays = self._prepare(row, sources)
            # Unique temporary names permit multiple readers/workers safely.
            import os
            import uuid
            temporary = cache_path.with_name(cache_path.stem + "." + uuid.uuid4().hex + ".tmp.npz")
            try:
                np.savez_compressed(temporary, **arrays)
                os.replace(str(temporary), str(cache_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
        with np.load(cache_path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata_json"]))
            sample = {"image": torch.from_numpy(arrays["image"].copy()),
                      "clinical": torch.from_numpy(hecktor_clinical(row)),
                      "time": torch.tensor(float(row["time"]), dtype=torch.float32),
                      "event": torch.tensor(float(row["recur_label"]), dtype=torch.float32),
                      "patient_id": row["patient_id"], "reference_mode": metadata["reference_mode"],
                      "mesh_vertex_count": metadata["mesh_vertex_count"],
                      "largest_component_fraction": metadata["largest_component_fraction"],
                      "metadata": metadata}
            if self.config["need_geometry"]:
                for name in ("vertices", "reference_origin", "reference_axes", "reference_length"):
                    sample[name] = torch.from_numpy(arrays[name].copy()).float()
                sample["faces"] = torch.from_numpy(arrays["faces"].copy()).long()
        if self.augment:
            # Intensity-only augmentation leaves reference axes and surface aligned.
            sample["image"] = (sample["image"] + .01 * torch.randn_like(sample["image"])).clamp(0, 1)
        return sample


def collate_batch(samples):
    if not samples:
        raise ValueError("Cannot collate empty samples")
    batch = {name: torch.stack([sample[name] for sample in samples])
             for name in ("image", "clinical", "time", "event")}
    for name in samples[0]:
        if name not in batch:
            batch[name] = [sample[name] for sample in samples]
    return batch


def hecktor_clinical(row):
    ts=['unknown','T0','T1','T2','T3','T4']; ns=['unknown','N0','N1','N2','N3']
    t=str(row.get('T-stage','')).strip().upper(); n=str(row.get('N-stage','')).strip().upper()
    return np.asarray(([float(x==t) for x in ts] if t in ts else [1.,0.,0.,0.,0.,0.])+([float(x==n) for x in ns] if n in ns else [1.,0.,0.,0.,0.]),dtype=np.float32)
