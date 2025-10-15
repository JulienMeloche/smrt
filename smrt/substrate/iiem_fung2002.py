"""
Implement the iiem rough substrate. See the documentation in :py:mod:`~smrt.interface.iiem_fung2002`

"""

# local import
#from smrt.interface.iiem_fung2002 import IIEM_Fung2002 as iIIEM_Fung2002
from iiem_fung2002 import IIEM_Fung2002 as iIIEM_Fung2002
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.GeometricalOptics
@substrate_from_interface(iIIEM_Fung2002)
class IIEM_Fung2002:
    pass