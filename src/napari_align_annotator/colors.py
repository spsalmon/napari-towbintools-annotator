# Palette of visually distinct colors shared across annotators.
CLASS_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def class_hex(classes, name):
    """Return the palette hex color for a class, by its index in ``classes``."""
    idx = classes.index(name)
    return CLASS_PALETTE[idx % len(CLASS_PALETTE)]


def hex_to_rgba_float(hex_str):
    """Convert a ``#RRGGBB`` string to an RGBA float tuple in [0, 1]."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b, 1.0)
