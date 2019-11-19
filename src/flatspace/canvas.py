from time import localtime, strftime
from typing import Tuple

import cv2 as cv
import numpy as np
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc as FourCC


def _as_int_tuple(arr):
    return tuple(int(x) for x in arr)


class Canvas():
    def __init__(
        self,
        resolution: Tuple,
        fps: int,
        px_per_unit: float = 1.0,
        preview: bool = True,
        render: bool = True,
    ):
        self.px_per_unit = px_per_unit
        self.resolution = _as_int_tuple(resolution)
        self.fps = float(fps)
        self.current_frame = self.new_frame()
        self.frame_no = 0

        # window and renderer
        self.preview = preview
        self.render = render
        filename = strftime('%Y%m%dT%H%M%S', localtime()) + ".mp4"
        if self.render:
            self.video = VideoWriter(f'./out/{filename}', FourCC(*"mp4v"), self.fps, self.resolution)
        self.title = f"flatspace preview ({filename})"

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        if self.preview:
            cv.destroyWindow(self.title)
        if self.render:
            self.video.release()

    @property
    def size(self):
        return self.px_to_units(self.resolution)

    def update(self):
        self.next_frame()
        self.check_exit()

    def next_frame(self):
        # output
        if self.render:
            print(f"rendering frame #{self.frame_no}", end="\r")
            self.video.write(self.current_frame)
        if self.preview:
            cv.imshow(self.title, self.current_frame)

        # next
        self.current_frame = self.new_frame()
        self.frame_no += 1

    def check_exit(self):
        # no preview -> no window to be closed
        if not self.preview:
            return

        key = cv.waitKey(1)
        if cv.getWindowProperty(self.title, 0) == -1 or key in {27, ord("q")}:  # window-x, esc or q
            self.__exit__()
            exit()

    def units_to_px(self, point):
        return np.array(self.resolution) / 2 + np.array((1, -1)) * self.px_per_unit * point

    def px_to_units(self, point):
        return (point - np.array(self.resolution) / 2) / (np.array((1, -1)) * self.px_per_unit)

    def new_frame(self):
        return np.ndarray(shape=(*self.resolution[::-1], 3), dtype="uint8")

    def fill(self, color):
        self.current_frame[:, :] = np.array(color[::-1], dtype="uint8")

    def draw_pixel(self, position, color):
        self.current_frame[self.units_to_px(position)] = np.array(color[::-1], dtype="uint8")

    def draw_rect(self, center, size, color):
        size = np.array(size) * self.px_per_unit
        center = self.units_to_px(center)
        cv.rectangle(
            self.current_frame,
            _as_int_tuple(center - size / 2),
            _as_int_tuple(center + size / 2),
            _as_int_tuple(color[::-1]),
            -1,
            16,
        )

    def draw_circle(self, center, radius, color):
        radius = np.array(radius) * self.px_per_unit
        center = self.units_to_px(center)
        cv.circle(
            self.current_frame,
            _as_int_tuple(center),
            int(radius),
            _as_int_tuple(color[::-1]),
            -1,
            16,
        )
