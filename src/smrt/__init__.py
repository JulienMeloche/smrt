# functions to be exported by default

import sys

if (sys.version_info[0] == 2) or ((sys.version_info[0] == 3) and (sys.version_info[1] <= 10)):
    raise RuntimeError("Python 3.10 and lower are not supported anymore")


from smrt.core import sensor as sensor
from smrt.core.error import SMRTError as SMRTError
from smrt.core.globalconstants import (
    PSU as PSU,
)
from smrt.core.globalconstants import (
    GHz as GHz,
)
from smrt.core.globalconstants import (
    cm as cm,
)
from smrt.core.globalconstants import (
    micron as micron,
)
from smrt.core.globalconstants import (
    mm as mm,
)
from smrt.core.model import (
    make_emmodel as make_emmodel,
)
from smrt.core.model import (
    make_model as make_model,
)
from smrt.core.model import (
    make_rtsolver as make_rtsolver,
)
from smrt.core.plugin import register_package as register_package
from smrt.core.result import open_result as open_result
from smrt.core.sensitivity_study import sensitivity_study as sensitivity_study
from smrt.inputs import sensor_list as sensor_list
from smrt.inputs.make_medium import (
    make_atmosphere as make_atmosphere,
)
from smrt.inputs.make_medium import (
    make_ice_column as make_ice_column,
)
from smrt.inputs.make_medium import (
    make_interface as make_interface,
)
from smrt.inputs.make_medium import (
    make_snow_layer as make_snow_layer,
)
from smrt.inputs.make_medium import (
    make_snowpack as make_snowpack,
)
from smrt.inputs.make_medium import (
    make_water_body as make_water_body,
)
from smrt.inputs.make_soil import make_soil as make_soil
from smrt.inputs.make_soil import make_soil_substrate as make_soil_substrate
