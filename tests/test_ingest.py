"""Tests for data ingestion utilities."""

from pathlib import Path

from PIL import Image

from newspapers.data.ingest import convert_jp2


class TestConvertJp2:
    """Test JPEG2000 → JPG/PNG conversion."""

    def test_converts_image(self, tmp_path: Path):
        # Create a small test image and save as PNG (stand-in for JP2)
        img = Image.new("RGB", (2000, 3000), color="white")
        src = tmp_path / "test_page.jp2"
        img.save(src, format="PNG")  # PIL writes PNG; JP2 needs extra codec

        out_dir = tmp_path / "output"
        jpg_path, png_path = convert_jp2(src, out_dir)

        assert jpg_path.exists()
        assert png_path.exists()
        assert jpg_path.suffix == ".jpg"
        assert png_path.suffix == ".png"

        # Low-res thumbnail should be smaller
        thumb = Image.open(jpg_path)
        assert max(thumb.size) <= 1280

    def test_output_dir_created(self, tmp_path: Path):
        img = Image.new("RGB", (100, 100), color="red")
        src = tmp_path / "page.jp2"
        img.save(src, format="PNG")

        out_dir = tmp_path / "nested" / "deep" / "out"
        convert_jp2(src, out_dir)

        assert out_dir.exists()
