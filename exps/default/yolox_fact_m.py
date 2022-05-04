import os

from yolox.exp import FacTExp
from yolox.models.transformers import *


class Exp(FacTExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        self.in_channels = (128, 256, 320)
        self.backbone = fact_tiny()
        return super().get_model()
