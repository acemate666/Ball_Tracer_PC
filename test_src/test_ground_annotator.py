import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageTk

from calibration.ground_annotator import AnnotatorApp


class GroundAnnotatorViewportTests(unittest.TestCase):
    def setUp(self):
        self.app = AnnotatorApp(world_points=[(1.0, 2.0, 0.0)])
        self.app.withdraw()
        self.addCleanup(self.app.destroy)
        for method, size in [('winfo_width', 400), ('winfo_height', 300)]:
            handle = patch.object(self.app.canvas, method, return_value=size)
            handle.start()
            self.addCleanup(handle.stop)
        image = Image.new('RGB', (256, 192), 'red')
        image.paste('lime', (128, 0, 256, 96))
        image.paste('blue', (0, 96, 128, 192))
        image.paste('yellow', (128, 96, 256, 192))
        self.app.pil_img = image

    def test_maximum_zoom_keeps_raster_bounded_by_viewport(self):
        self.app.scale = 50.0
        self.app.offset_x = 200 - 128 * 50
        self.app.offset_y = 150 - 96 * 50
        self.app._full_redraw()
        self.assertEqual(self.app._cached_tk_img.width(), 400)
        self.assertEqual(self.app._cached_tk_img.height(), 300)
        self.assertEqual(self.app.canvas.coords(self.app._img_on_canvas), [0.0, 0.0])

    def test_drag_renders_newly_visible_pixels(self):
        self.app.scale = 2.0
        self.app.offset_x = -256.0
        self.app._full_redraw()
        before = ImageTk.getimage(self.app._cached_tk_img).convert('RGB')
        self.assertEqual(before.getpixel((50, 50)), (0, 255, 0))
        self.app._on_drag_start(SimpleNamespace(x=10, y=10))
        self.app._on_drag_move(SimpleNamespace(x=266, y=10))
        self.app._full_redraw()
        after = ImageTk.getimage(self.app._cached_tk_img).convert('RGB')
        self.assertEqual(after.getpixel((50, 50)), (255, 0, 0))

    def test_zoom_preserves_cursor_anchor_and_annotation_coordinates(self):
        self.app.scale = 2.0
        self.app.offset_x = -120.0
        self.app.offset_y = -70.0
        wheel = SimpleNamespace(x=173, y=111, delta=120)
        anchor = self.app._canvas_to_pixel(wheel.x, wheel.y)
        self.app._on_scroll(wheel)
        timer = self.app._redraw_timer
        self.app._on_scroll(wheel)
        self.assertEqual(timer, self.app._redraw_timer)
        restored = self.app._canvas_to_pixel(wheel.x, wheel.y)
        self.assertAlmostEqual(anchor[0], restored[0])
        self.assertAlmostEqual(anchor[1], restored[1])
        self.app._full_redraw()
        self.app.annotating = True
        cx, cy = self.app._pixel_to_canvas(140, 80)
        self.app._on_click(SimpleNamespace(x=cx, y=cy))
        self.assertEqual(self.app.annotations, [{'pixel': (140, 80), 'world': (1.0, 2.0, 0.0)}])
        circle = self.app.canvas.coords(self.app._marker_ids[0][0])
        self.assertAlmostEqual((circle[0] + circle[2]) / 2, cx)
        self.assertAlmostEqual((circle[1] + circle[3]) / 2, cy)


if __name__ == '__main__':
    unittest.main()
