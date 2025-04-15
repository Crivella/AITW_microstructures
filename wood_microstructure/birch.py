"""Ray cells"""

from .microstructure import WoodMicrostructure


class BirchMicrostructure(WoodMicrostructure):
    local_distortion_cutoff = 200
