"""Provide the microstructure for independent cylinders. "Independent" means that the cylinders are
distributed completely randomly in the medium, even overlapping each other. This is why this representation only
applies to low fractional volume.

"""

from smrt.microstructure_model.independent_disk import IndependentDisk


class IndependentCylinder(IndependentDisk):
    pass
