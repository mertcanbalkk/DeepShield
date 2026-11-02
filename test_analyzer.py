import unittest
from io import BytesIO
import numpy as np
from PIL import Image
from image_analyzer import compare_to_original, inspect_without_original, load_rgb, extract_metadata


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.original = rng.integers(50, 160, (400, 600, 3), dtype=np.uint8)

    def test_exact_same(self):
        result = compare_to_original(self.original, self.original)
        self.assertEqual(result["difference_percent"], 0)
        self.assertEqual(result["regions"], [])

    def test_known_square_edit(self):
        changed = self.original.copy()
        changed[160:230, 240:350] = [255, 10, 30]
        result = compare_to_original(changed, self.original)
        self.assertGreater(result["difference_percent"], 0)
        self.assertTrue(any(
            r["x"] < 250 and r["x"]+r["width"] > 340 and
            r["y"] < 170 and r["y"]+r["height"] > 220
            for r in result["regions"]
        ))

    def test_reference_resize_same_aspect(self):
        import cv2
        bigger = cv2.resize(self.original, (900, 600), interpolation=cv2.INTER_LINEAR)
        result = compare_to_original(bigger, self.original)
        self.assertIn("yeniden boyutlandırma", result["alignment"])

    def test_indicator_no_certainty(self):
        result = inspect_without_original(self.original, "example.png")
        self.assertNotIn("status", result)
        self.assertIn("kesin", result["conclusion"])
        self.assertEqual(result["ela"].shape, (400,600))

    def test_highlight_obvious_in_mask(self):
        changed = self.original.copy()
        changed[100:190, 140:240] = [255, 15, 30]
        result = compare_to_original(changed, self.original)
        self.assertEqual(result["marked"].shape, changed.shape)
        self.assertEqual(result["mask_view"].shape, changed.shape)
        self.assertGreater(int(np.count_nonzero(result["mask_view"])), 0)

    def test_image_metadata(self):
        buf = BytesIO()
        Image.fromarray(self.original).save(buf, format="PNG")
        data = buf.getvalue()
        self.assertEqual(load_rgb(data).shape, self.original.shape)
        self.assertEqual(extract_metadata(data)["format"], "PNG")

if __name__ == "__main__":
    unittest.main()
