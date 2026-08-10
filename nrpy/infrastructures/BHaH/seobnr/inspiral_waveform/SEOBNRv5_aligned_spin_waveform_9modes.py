"""
Set up C function library for the SEOBNRv5 aligned-spin inspiral waveform (9 modes).

This is an example-specific variant of ``SEOBNRv5_aligned_spin_waveform_higher_mode`` that
emits every ``(l,m)`` mode with a ``delta_lm`` implemented (except (8,8), whose ``delta_lm`` is
identically zero): (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (6,6), (7,7).

Unlike the shared higher-mode generator, the ``gamma_lm`` declarations and ``STRAIN`` stores are
generated from the explicit mode list so nothing is hardcoded to a fixed mode count.

Authors: Siddharth Mahesh
        sm0193 **at** mix **dot** wvu **dot** edu
        Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.equations.seobnr.SEOBNRv5_aligned_spin_waveform_quantities as SEOBNRv5_wf
import nrpy.helpers.parallel_codegen as pcg

# The (l, m) modes with a delta_lm implemented, excluding (8,8) (delta_88 == 0).
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5), (6, 6), (7, 7)]


def register_CFunction_SEOBNRv5_aligned_spin_waveform() -> (
    Union[None, pcg.NRPyEnv_type]
):
    """
    Register CFunction for calculating multiple SEOBNRv5 aligned-spin inspiral waveform modes.

    Computes the modes (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (6,6), (7,7) for a single
    timestep and stores them in the inspiral_modes array. The special amplitude coefficients
    c_21/c_43/c_55 are read from commondata; leaving them at their default of 0 disables them.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    wf = SEOBNRv5_wf.SEOBNRv5_aligned_spin_waveform_quantities(
        apply_special_amplitude_coefficients=True
    )
    hlms_dict = wf.strain()
    hlms = []
    hlms_labels = []
    khatm = []
    khatm_labels = []

    seen_m = set()  # preserves insertion order via the lists, uniqueness via this set
    for l, m in MODES:
        key = f"({l} , {m})"
        hlms.append(hlms_dict[key])
        hlms_labels.append(f"const double complex h{l}{m}")
        if m not in seen_m:
            khatm.append(wf.khat[m])
            khatm_labels.append(f"const REAL khat{m}")
            seen_m.add(m)

    # We are going to be doing this twice;
    # once for the fine dynamics and once for the coarse.
    h_code = ccg.c_codegen(
        hlms,
        hlms_labels,
        verbose=False,
        include_braces=False,
        fp_type="double complex",
        fp_type_alias="COMPLEX",
    )
    khat_code = ccg.c_codegen(
        khatm,
        khatm_labels,
        verbose=False,
        include_braces=False,
        cse_varprefix="khat",
    )

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = """
Calculates SEOBNRv5 aligned-spin inspiral waveform modes
(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (6,6), (7,7) for a single timestep
and stores them in the inspiral_modes array.

@param dynamics - Array of dynamical variables.
@param commondata - Common data structure containing the model parameters.
@param inspiral_modes - Output array of the SEOBNRv5 inspiral waveform modes.
"""
    cfunc_type = "void"
    prefunc = "#include<complex.h>"
    name = "SEOBNRv5_aligned_spin_waveform"
    params = "REAL *restrict dynamics, commondata_struct *restrict commondata, double complex *inspiral_modes"
    body = ""
    for l, m in MODES:
        body += f"COMPLEX gamma_{l}{m};\n"
    body += """const REAL m1 = commondata->m1;
const REAL m2 = commondata->m2;
const REAL chi1 = commondata->chi1;
const REAL chi2 = commondata->chi2;
const REAL c_21 = commondata->c_21;
const REAL c_43 = commondata->c_43;
const REAL c_55 = commondata->c_55;
const REAL phi = dynamics[PHI];
const REAL pphi = dynamics[PPHI];
const REAL Hreal = dynamics[H];
const REAL Omega = dynamics[OMEGA];
const REAL Omega_circ = dynamics[OMEGA_CIRC];
//compute
"""
    body += khat_code
    for l, m in MODES:
        body += f"""
    gamma_{l}{m} = SEOBNRv5_aligned_spin_gamma_wrapper({l} + 1, -2.*khat{m});
"""

    body += h_code
    body += "\n"
    for l, m in MODES:
        body += f"inspiral_modes[STRAIN{l}{m} - 1] = h{l}{m};\n"
    cfc.register_CFunction(
        subdirectory="inspiral_waveform",
        includes=includes,
        desc=desc,
        prefunc=prefunc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return pcg.NRPyEnv()
