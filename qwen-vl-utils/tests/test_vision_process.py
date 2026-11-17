import io

from PIL import Image, ImageChops, ImageDraw

from qwen_vl_utils import fetch_image
from qwen_vl_utils.vision_process import to_rgb


def _png_roundtrip(im, **kwargs):
    buf = io.BytesIO()
    im.save(buf, "PNG", **kwargs)
    return Image.open(io.BytesIO(buf.getvalue()))


def _black_outline_on_transparent(size=224):
    """Black rectangle outline on a fully transparent background (issue #2138 repro)."""
    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(base).rectangle([56, 56, 168, 168], outline=(0, 0, 0, 255), width=8)
    return base


def _palette_with_transparency(base):
    pal = Image.new("P", base.size, 0)
    pal.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)  # colour 0 = transparent, colour 1 = black
    pal.paste(1, mask=base.getchannel("A"))
    return pal


def _legacy_to_rgb(pil_image):
    """Pre-fix behaviour: only RGBA is composited onto white."""
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")


def _images_equal(a, b):
    return a.mode == b.mode and a.size == b.size and ImageChops.difference(a, b).getbbox() is None


def _transparent_cases():
    base = _black_outline_on_transparent()
    return {
        "RGBA": _png_roundtrip(base),
        "LA": _png_roundtrip(base.convert("LA")),
        "P": _png_roundtrip(_palette_with_transparency(base), transparency=0),
    }


def test_to_rgb_composites_transparent_modes_onto_white():
    for name, im in _transparent_cases().items():
        assert im.mode == name
        out = to_rgb(im).convert("L")
        assert out.getpixel((10, 10)) == 255, f"{name} background should be white, got {out.getpixel((10, 10))}"
        assert out.getpixel((112, 60)) == 0, f"{name} line should be black, got {out.getpixel((112, 60))}"


def test_fetch_image_composites_transparent_modes_onto_white():
    for name, im in _transparent_cases().items():
        out = fetch_image({"image": im}).convert("L")
        assert out.getpixel((10, 10)) == 255, f"{name} background should be white, got {out.getpixel((10, 10))}"
        assert out.getpixel((112, 60)) == 0, f"{name} line should be black, got {out.getpixel((112, 60))}"


def test_to_rgb_rgba_and_opaque_images_match_previous_behavior():
    rgba = _png_roundtrip(_black_outline_on_transparent())
    rgba.putpixel((20, 20), (10, 20, 30, 128))
    assert _images_equal(to_rgb(rgba), _legacy_to_rgb(rgba))

    rgb = Image.new("RGB", (32, 32), (12, 34, 56))
    rgb.putpixel((1, 1), (7, 8, 9))
    assert _images_equal(to_rgb(rgb), _legacy_to_rgb(rgb))

    gray = Image.new("L", (32, 32), 40)
    gray.putpixel((1, 1), 200)
    assert _images_equal(to_rgb(gray), _legacy_to_rgb(gray))

    pal = Image.new("P", (32, 32), 0)
    pal.putpalette([10, 20, 30, 40, 50, 60] + [0] * 762)
    pal.putpixel((1, 1), 1)
    pal = _png_roundtrip(pal)
    assert "transparency" not in pal.info
    assert _images_equal(to_rgb(pal), _legacy_to_rgb(pal))
