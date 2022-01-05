from enum import IntEnum, unique
from constants import IMAGES_DIR
import os


@unique
class Marker(IntEnum):
    LEFT = 2
    RIGHT = 1
    IDLE = 3

    @property
    def image_path(self):
        return os.path.join(IMAGES_DIR, f'{self.name}.png')

    @classmethod
    def all(cls):
        return [stim.value for stim in cls]
