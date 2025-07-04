"""
This package contains the different electromagnetic (EM) models that compute the scattering and absorption coefficients
and the phase function in a given layer. The computation of the inter-layer propagation is done by the
`smrt.rtsolver` package.

The EM models differ in many aspects, one of which is the constraint on the microstructure model
they can be used with. The:py:mod:`smrt.emmodel.iba` model can use any
microstructure model that defines autocorrelation functions (or its FT). In contrast, others such as
`smrt.emmodel.dmrt_shortrange` are bound to the:py:mod:`smrt.microstructuremodel.sticky_hard_spheres` microstructure 
for theoretical reasons.

The selection of the EM model is done with the :py:mod:`smrt.core.model.make_model` function.

Note:  **For developers**

    To implement a new scattering formulation / phase function, we recommend to start from an existing module, probably rayleigh.py is the simplest.
    Copy this file to `myscatteringtheory.py` or any meaningful name. It can be directly used with :py:func:`smrt.core.model.make_model` function as follows::

        m = make_model("myscatteringtheory", "dort")

    Note that if the file is not in the emmodels directory, you must explicitly import the module and pass it 
    to make_model as a module object (instead of a string).

    An emmodel model must define:
        - ks and ka attributes/properties
        - ke() and effective_permittivity() methods
        - at least one of the phase and ft_even_phase methods (both is better).

    For the details it is recommended to contact the authors as the calling arguments and required methods may change from time to time.
"""
