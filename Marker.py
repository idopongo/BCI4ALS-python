from enum import IntEnum, unique
from constants import IMAGES_DIR
import os

@unique
class Marker(IntEnum):
    RIGHT = 1
    LEFT = 2
    IDLE = 3
    STOP = 4

    @property
    def image_path(self):
        return os.path.join(IMAGES_DIR, f'{self.name}.png')

    @classmethod
    def stimuli(cls):
        return [cls.RIGHT, cls.LEFT, cls.IDLE]
