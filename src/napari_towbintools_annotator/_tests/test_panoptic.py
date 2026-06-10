from napari_towbintools_annotator.colors import (
    CLASS_PALETTE,
    class_hex,
    hex_to_rgba_float,
)


def test_hex_to_rgba_float_white():
    assert hex_to_rgba_float("#ffffff") == (1.0, 1.0, 1.0, 1.0)


def test_hex_to_rgba_float_strips_hash_and_scales():
    r, g, b, a = hex_to_rgba_float("#4C72B0")
    assert a == 1.0
    assert abs(r - 0x4C / 255) < 1e-9
    assert abs(g - 0x72 / 255) < 1e-9
    assert abs(b - 0xB0 / 255) < 1e-9


def test_class_hex_indexes_palette_and_wraps():
    classes = list(range(len(CLASS_PALETTE) + 2))
    assert class_hex(classes, 0) == CLASS_PALETTE[0]
    assert class_hex(classes, len(CLASS_PALETTE)) == CLASS_PALETTE[0]
