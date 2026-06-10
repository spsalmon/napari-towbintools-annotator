import numpy as np
from skimage.measure import regionprops


def nearest_class_id(color, id_to_color):
    """Return the class id whose RGBA color is closest to ``color``."""
    target = np.asarray(color, dtype=float)
    best_id, best_dist = -1, float("inf")
    for class_id, class_color in id_to_color.items():
        dist = np.sum((target - np.asarray(class_color, dtype=float)) ** 2)
        if dist < best_dist:
            best_dist, best_id = dist, class_id
    return best_id


def points_to_rows(
    points, face_colors, label_data, id_to_color, id_to_name, plane_axis=None
):
    """Convert annotation points + colors into per-instance annotation rows.

    Each point is rounded to integer coordinates, used to read the label value
    under it, and its color is matched to the nearest class. Points outside the
    label array are skipped. In 3D (``plane_axis`` set) the first-axis index is
    recorded under that column name.
    """
    rows = []
    shape = label_data.shape
    for point, color in zip(points, face_colors):
        index = tuple(int(round(coord)) for coord in point)
        if len(index) != len(shape):
            continue
        if any(i < 0 or i >= s for i, s in zip(index, shape)):
            continue
        label_value = int(label_data[index])
        class_id = nearest_class_id(color, id_to_color)
        class_name = id_to_name.get(class_id, "unknown")
        if plane_axis is not None:
            row = {
                plane_axis: index[0],
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        else:
            row = {
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        rows.append(row)
    return rows


def _centroid(mask):
    props = regionprops(mask.astype(int))
    if not props:
        return None
    return props[0].centroid


def rows_to_points(annotations_df, label_data, id_to_color, plane_axis=None):
    """Convert annotation rows back into ``(point_coords, rgba)`` placements.

    For each row the label's centroid is used as the point location. Rows with
    an unknown class id, or whose label is absent from the (plane of the) label
    array, are skipped.
    """
    placements = []
    if plane_axis is not None:
        for plane in annotations_df[plane_axis].unique():
            plane = int(plane)
            if plane < 0 or plane >= label_data.shape[0]:
                continue
            plane_df = annotations_df[annotations_df[plane_axis] == plane]
            for _, row in plane_df.iterrows():
                color = id_to_color.get(int(row["ClassID"]))
                if color is None:
                    continue
                mask = label_data[plane] == int(row["Label"])
                if not mask.any():
                    continue
                centroid = _centroid(mask)
                if centroid is None:
                    continue
                placements.append(
                    (np.array([plane, centroid[0], centroid[1]]), color)
                )
    else:
        for _, row in annotations_df.iterrows():
            color = id_to_color.get(int(row["ClassID"]))
            if color is None:
                continue
            mask = label_data == int(row["Label"])
            if not mask.any():
                continue
            centroid = _centroid(mask)
            if centroid is None:
                continue
            placements.append(
                (np.array([centroid[0], centroid[1]]), color)
            )
    return placements
