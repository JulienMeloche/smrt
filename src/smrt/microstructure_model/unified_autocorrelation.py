# local import
from smrt.core.globalconstants import DENSITY_OF_ICE
from smrt.microstructure_model.autocorrelation import Autocorrelation


class UnifiedAutocorrelation(Autocorrelation):
    args = ["frac_volume", "porod_length", "polydispersity"]
    optional_args = {}

    def compute_ssa(self):
        """Compute the ssa for the given porod_length."""
        return 3 * (1 - self.frac_volume) / (DENSITY_OF_ICE * self.porod_length)
