"""Recover cross-plane instance identity from plane-wise segmentations.

Panoptic z-stacks are labelled independently in each plane: the same nucleus
carries a different label value in every slice. These helpers stitch adjacent
planes by intersection-over-union to recover which labels belong to the same
object, so that annotating a nucleus once can be propagated through the stack.

The original label values are never modified. Stitching only derives a mapping;
the segmentation on disk stays the ground truth.
"""

import hashlib
import os

import numpy as np
from scipy import ndimage as ndi

# Above this label value, building a lookup table costs more than sorting, so
# fall back to np.unique. Ordinary uint8/uint16 masks stay well under it.
_MAX_LUT_LABEL = 1_000_000


def _relabel_contiguous(plane):
    """Return ``plane`` relabelled to ``0..N`` plus the original label values.

    The IoU matrix below is sized by the largest label present, so a plane
    holding sparse values (e.g. a single object labelled 65535) would otherwise
    allocate an enormous array. Compacting first bounds that by the object
    count. ``values[i]`` is the original label for compact label ``i``, with
    index 0 reserved for background.

    Uses a counting pass rather than a sort: planes are millions of pixels and
    this runs once per plane per load.
    """
    flat = plane.ravel()
    peak = int(flat.max()) if flat.size else 0

    if peak <= _MAX_LUT_LABEL:
        present = np.nonzero(np.bincount(flat))[0]
        lookup = np.zeros(peak + 1, dtype=np.int64)
    else:  # pragma: no cover - pathological label ranges
        present = np.unique(flat)
        lookup = None

    if len(present) and present[0] == 0:
        values = present
        compact_ids = np.arange(len(present), dtype=np.int64)
    else:
        values = np.concatenate(([0], present))
        compact_ids = np.arange(1, len(present) + 1, dtype=np.int64)

    if np.array_equal(values, np.arange(len(values))):
        # Already 0..N — the common case for segmenter output. Remapping
        # would be an identity gather over millions of pixels, so skip it.
        return plane, values

    if lookup is None:  # pragma: no cover - pathological label ranges
        compact = np.searchsorted(values, flat).reshape(plane.shape)
    else:
        lookup[present] = compact_ids
        compact = lookup[flat].reshape(plane.shape)
    return compact, values


def _compact_planes(masks):
    """Compact every plane once, for all three consumers below."""
    compacted, values = [], []
    for plane in masks:
        plane_compact, plane_values = _relabel_contiguous(plane)
        compacted.append(plane_compact)
        values.append(plane_values)
    counts = [len(v) - 1 for v in values]
    return compacted, values, counts


def _iou_matrix(a, b, n_a, n_b):
    """Pairwise IoU between the objects of two compacted planes.

    Returns an ``(n_a, n_b)`` array; background is excluded from both axes.
    """
    # Widen before combining: a uint16 plane with a few hundred objects would
    # otherwise wrap silently on the multiply and mis-stitch.
    pairs = a.ravel().astype(np.int64) * (n_b + 1) + b.ravel()
    overlap = np.bincount(pairs, minlength=(n_a + 1) * (n_b + 1)).reshape(
        n_a + 1, n_b + 1
    )
    area_a = overlap.sum(axis=1, keepdims=True)
    area_b = overlap.sum(axis=0, keepdims=True)
    union = area_a + area_b - overlap
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, overlap / np.maximum(union, 1), 0.0)
    return iou[1:, 1:]


def _stitch_lookups(compacted, counts, threshold):
    """Per-plane arrays mapping compact label -> global instance id.

    ``lookups[z][c]`` is the instance of compact label ``c`` in plane ``z``,
    with index 0 held at 0 for background.
    """
    lookups = [np.arange(counts[0] + 1, dtype=np.int64)]
    running_max = counts[0]

    for i in range(len(compacted) - 1):
        n_current, n_previous = counts[i + 1], counts[i]

        if n_current == 0:
            lookups.append(np.zeros(1, dtype=np.int64))
            continue

        if n_previous == 0:
            assigned = np.arange(
                running_max + 1, running_max + n_current + 1, dtype=np.int64
            )
            running_max += n_current
        else:
            iou = _iou_matrix(
                compacted[i + 1], compacted[i], n_current, n_previous
            )
            iou[iou < threshold] = 0.0
            # Keep only the best claim on each previous object.
            iou[iou < iou.max(axis=0)] = 0.0
            best = iou.argmax(axis=1) + 1
            assigned = lookups[i][best]
            unmatched = np.nonzero(iou.max(axis=1) == 0.0)[0]
            assigned[unmatched] = np.arange(
                running_max + 1,
                running_max + len(unmatched) + 1,
                dtype=np.int64,
            )
            running_max += len(unmatched)

        lookups.append(np.concatenate(([0], assigned)).astype(np.int64))

    return lookups


def stitch_planes(masks, threshold=0.25):
    """Relabel a ``[Z, Y, X]`` stack so one object keeps one id across planes.

    Each plane is matched against the previous one; an object linked to a
    previous object above ``threshold`` inherits its id, otherwise it starts a
    new one. Equivalent to ``cellpose.utils.stitch3D``, vendored to avoid a
    cellpose runtime dependency.

    The result is ``int64``, so the running id count cannot overflow the input
    mask's dtype.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(
            f"stitch_planes expects a 3D [Z, Y, X] stack, got {masks.ndim}D."
        )

    compacted, _, counts = _compact_planes(masks)
    lookups = _stitch_lookups(compacted, counts, threshold)

    stitched = np.zeros(masks.shape, dtype=np.int64)
    for z, (plane, lookup) in enumerate(zip(compacted, lookups, strict=True)):
        stitched[z] = lookup[plane]
    return stitched


def _centroids_from_compacted(compacted, values):
    """Centroids keyed by ``(z, original_label)``.

    Each centroid is computed inside its label's bounding box rather than by
    scanning the whole plane, which matters because a stack can hold thousands
    of ``(z, label)`` pairs once annotations are propagated.
    """
    table = {}
    for z, (plane, plane_values) in enumerate(
        zip(compacted, values, strict=True)
    ):
        for compact_label, box in enumerate(ndi.find_objects(plane), start=1):
            if box is None:
                continue
            local_y, local_x = ndi.center_of_mass(plane[box] == compact_label)
            table[(z, int(plane_values[compact_label]))] = (
                local_y + box[0].start,
                local_x + box[1].start,
            )
    return table


def centroid_table(masks):
    """Map every ``(z, label)`` in a ``[Z, Y, X]`` stack to its centroid."""
    compacted, values, _ = _compact_planes(np.asarray(masks))
    return _centroids_from_compacted(compacted, values)


class InstanceIndex:
    """Cross-plane identity for one segmentation stack.

    Instance ids exist only to group planes together. Every ``Label`` written
    to disk is still an original value read from the segmentation file.
    """

    def __init__(
        self,
        plane_label_to_instance,
        instance_to_planes,
        centroids,
        threshold,
        masks=None,
    ):
        self.plane_label_to_instance = plane_label_to_instance
        self.instance_to_planes = instance_to_planes
        self.centroids = centroids
        self.threshold = threshold
        self.masks = masks

    def instance_at(self, z, y, x):
        """Instance under a click, or ``None`` for background/out of bounds."""
        if self.masks is None:
            return None
        z, y, x = int(z), int(y), int(x)
        shape = self.masks.shape
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            return None
        label = int(self.masks[z, y, x])
        if label == 0:
            return None
        return self.plane_label_to_instance.get((z, label))

    def planes_of(self, instance_id):
        """``{z: original_label}`` for an instance."""
        return self.instance_to_planes.get(instance_id, {})

    def centroid(self, z, label):
        return self.centroids.get((int(z), int(label)))


def build_instance_index(masks, threshold=0.25):
    """Stitch a stack and package the resulting identity mapping."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(
            f"build_instance_index expects a 3D stack, got {masks.ndim}D."
        )

    compacted, values, counts = _compact_planes(masks)
    lookups = _stitch_lookups(compacted, counts, threshold)

    plane_label_to_instance = {}
    instance_to_planes = {}
    for z, (plane_values, lookup) in enumerate(
        zip(values, lookups, strict=True)
    ):
        # Compact label c is original label plane_values[c]; the stitch pass
        # already resolved its instance, so no pixel lookup is needed.
        for compact_label in range(1, len(plane_values)):
            label = int(plane_values[compact_label])
            instance = int(lookup[compact_label])
            plane_label_to_instance[(z, label)] = instance
            instance_to_planes.setdefault(instance, {})[z] = label

    return InstanceIndex(
        plane_label_to_instance=plane_label_to_instance,
        instance_to_planes=instance_to_planes,
        centroids=_centroids_from_compacted(compacted, values),
        threshold=threshold,
        masks=masks,
    )


def instance_volume(index):
    """Render an index's instance ids as a ``[Z, Y, X]`` array.

    Display only. The array lives in memory and the segmentation on disk is
    untouched — it exists so one nucleus can be shown in one colour through
    the stack. Built from the stored mapping rather than by stitching again,
    so an index served from cache renders just as cheaply.
    """
    if index.masks is None:
        raise ValueError(
            "instance_volume needs the masks the index was built from."
        )
    masks = np.asarray(index.masks)

    by_plane = {}
    for (z, label), instance in index.plane_label_to_instance.items():
        by_plane.setdefault(z, []).append((label, instance))

    volume = np.zeros(masks.shape, dtype=np.int32)
    for z, pairs in by_plane.items():
        plane = masks[z]
        lookup = np.zeros(int(plane.max()) + 1, dtype=np.int32)
        for label, instance in pairs:
            lookup[label] = instance
        volume[z] = lookup[plane]
    return volume


def instance_counts(index):
    """``(plane objects, instances)`` — how much the stitch collapsed."""
    return (
        len(index.plane_label_to_instance),
        len(index.instance_to_planes),
    )


def _cache_key(segmentation_path, threshold):
    """Identity of a stitch result: which file, in which state, at which
    threshold. A re-segmented file or a retuned threshold simply misses."""
    stat = os.stat(segmentation_path)
    material = "|".join(
        (
            os.path.abspath(segmentation_path),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            f"{float(threshold):.6f}",
        )
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def save_index_to_cache(cache_dir, segmentation_path, index):
    """Write an index to the cache. Never raises: caching is best-effort."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        keys = sorted(index.plane_label_to_instance)
        np.savez_compressed(
            os.path.join(
                cache_dir, _cache_key(segmentation_path, index.threshold)
            ),
            z=np.array([k[0] for k in keys], dtype=np.int64),
            label=np.array([k[1] for k in keys], dtype=np.int64),
            instance=np.array(
                [index.plane_label_to_instance[k] for k in keys],
                dtype=np.int64,
            ),
            cy=np.array(
                [index.centroids[k][0] for k in keys], dtype=np.float64
            ),
            cx=np.array(
                [index.centroids[k][1] for k in keys], dtype=np.float64
            ),
            threshold=np.array([index.threshold], dtype=np.float64),
        )
    except OSError:
        pass


def load_cached_index(cache_dir, segmentation_path, threshold, masks):
    """Return a cached index, or ``None`` on any miss.

    A missing, unreadable, or corrupt cache entry is a miss rather than an
    error — the caller recomputes.
    """
    try:
        path = os.path.join(
            cache_dir, _cache_key(segmentation_path, threshold) + ".npz"
        )
        with np.load(path) as data:
            z = data["z"]
            label = data["label"]
            instance = data["instance"]
            cy = data["cy"]
            cx = data["cx"]
            stored_threshold = float(data["threshold"][0])
    except (OSError, KeyError, ValueError, EOFError):
        return None

    plane_label_to_instance = {}
    instance_to_planes = {}
    centroids = {}
    for zi, li, ii, y, x in zip(z, label, instance, cy, cx, strict=True):
        key = (int(zi), int(li))
        plane_label_to_instance[key] = int(ii)
        instance_to_planes.setdefault(int(ii), {})[int(zi)] = int(li)
        centroids[key] = (float(y), float(x))

    return InstanceIndex(
        plane_label_to_instance=plane_label_to_instance,
        instance_to_planes=instance_to_planes,
        centroids=centroids,
        threshold=stored_threshold,
        masks=masks,
    )


def get_instance_index(segmentation_path, masks, threshold, cache_dir=None):
    """Index for a segmentation, from cache when possible.

    The first visit to a stack pays the stitching cost; later visits are
    served from disk.
    """
    if cache_dir is not None:
        cached = load_cached_index(
            cache_dir, segmentation_path, threshold, masks
        )
        if cached is not None:
            return cached

    index = build_instance_index(masks, threshold=threshold)
    if cache_dir is not None:
        save_index_to_cache(cache_dir, segmentation_path, index)
    return index
