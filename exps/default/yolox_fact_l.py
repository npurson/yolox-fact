import os

from yolox.exp import FacTExp
from yolox.models.transformers import *


class Exp(FacTExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        self.in_channels = (192, 384, 768)
        self.backbone = fact_small()
        return super().get_model()
