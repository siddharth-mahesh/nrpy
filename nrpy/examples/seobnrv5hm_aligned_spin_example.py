"""
Set up a complete C code project that computes the SEOBNRv5 aligned-spin inspiral higher modes.

Unlike ``seobnrv5_aligned_spin_inspiral.py`` (which computes the (2,2) mode and attaches a
merger-ringdown), this example stops at the inspiral and outputs every ``(l,m)`` mode for which
the SEOBNRv5 aligned-spin waveform has a ``delta_lm`` implemented, excluding (8,8) whose
``delta_lm`` is identically zero. The output modes are:

    (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (6,6), (7,7)

These are the "raw" inspiral modes computed directly from the dynamics: no NQC corrections, no
special amplitude coefficients, and no merger/BOB attachment. The modes are resampled onto a
uniform time grid before being printed.

Authors: Siddharth Mahesh
        sm0193 **at** mix **dot** wvu **dot** edu
        Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import os

#########################################################
# STEP 1: Import needed Python modules, then set codegen
#         and compile-time parameters.
import shutil
from pathlib import Path

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
import nrpy.params as par
from nrpy.infrastructures import BHaH
from nrpy.infrastructures.BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_waveform_9modes import (
    MODES,
)

par.set_parval_from_str("Infrastructure", "BHaH")
enable_parallel_codegen = True
par.set_parval_from_str("enable_parallel_codegen", enable_parallel_codegen)

# Code-generation-time parameters:
project_name = "seobnrv5hm_aligned_spin"
project_dir = os.path.join("project", project_name)

# First clean the project directory, if it exists.
shutil.rmtree(project_dir, ignore_errors=True)

# Development flags (NOT command-line-tunable)
# Flag to output the commondata struct to a file.
output_commondata_flag = False
# Flag to output the waveform using a print statement.
output_waveform_flag = True

#########################################################
# STEP 2: Declare core C functions & register each to
#         cfc.CFunction_dict["function_name"]


def register_CFunction_main_c(
    output_waveform: bool = True,
    output_commondata: bool = False,
) -> None:
    """
    Generate a C main() function that computes the SEOBNRv5 aligned-spin inspiral higher modes.

    :param output_waveform: Flag to enable/disable printing the waveform.
    :param output_commondata: Flag to enable/disable outputting the commondata struct to a binary file.
    """
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = """-={ main() function }=-
Step 1.a: Set each commondata CodeParameter to default.
Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
Step 2: Compute the SEOBNRv5 aligned-spin inspiral higher modes."""
    cfunc_type = "int"
    name = "main"
    params = "int argc, const char *argv[]"
    body = r"""  commondata_struct commondata; // commondata contains parameters common to all grids.
// Step 1.a: Set each commondata CodeParameter to default.
commondata_struct_set_to_default(&commondata);
// Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
cmdline_input_and_parfile_parser(&commondata, argc, argv);
// Step 1.c: Overwrite default values of m1, m2, a6, and dSO.
SEOBNRv5_aligned_spin_coefficients(&commondata);
// Step 2.a: Compute SEOBNRv5 conservative initial conditions.
SEOBNRv5_aligned_spin_initial_conditions_conservative(&commondata);
// Step 2.b: Compute SEOBNRv5 dissipative initial conditions.
SEOBNRv5_aligned_spin_initial_conditions_dissipative(&commondata);
// Step 3: Run the ODE integration.
SEOBNRv5_aligned_spin_ode_integration(&commondata);
// Step 4: Generate the inspiral higher modes from the dynamics.
SEOBNRv5_aligned_spin_waveform_from_dynamics(&commondata);
// Step 5: Resample all modes onto a uniform time grid.
const REAL dT = commondata.dt/(commondata.total_mass*4.925490947641266978197229498498379006e-6);
SEOBNRv5_aligned_spin_interpolate_modes(&commondata, dT);
"""
    if output_waveform:
        strain_columns = ", ".join(
            f"creal(commondata.waveform_inspiral[IDX_WF(i,STRAIN{l}{m})]), "
            f"cimag(commondata.waveform_inspiral[IDX_WF(i,STRAIN{l}{m})])"
            for l, m in MODES
        )
        fmt = "%.15e " + " ".join(["%.15e %.15e"] * len(MODES))
        body += rf"""
// Step 6: Print the resulting inspiral modes: time followed by Re/Im of each mode.
for (size_t i = 0; i < commondata.nsteps_inspiral; i++) {{
    printf("{fmt}\n", creal(commondata.waveform_inspiral[IDX_WF(i,TIME)]), {strain_columns});
}}
"""
    if output_commondata:
        body += r"""
commondata_io(&commondata, "commondata.bin");
"""
    body += r"""
free(commondata.dynamics_low);
free(commondata.dynamics_fine);
free(commondata.dynamics_raw);
free(commondata.waveform_low);
free(commondata.waveform_fine);
free(commondata.waveform_inspiral);
return 0;
"""
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
    )


# register utilities needed by the waveform code
BHaH.seobnr.utils.commondata_io.register_CFunction_commondata_io()
BHaH.seobnr.utils.handle_gsl_return_status.register_CFunction_handle_gsl_return_status()
BHaH.seobnr.utils.SEOBNRv5_aligned_spin_unwrap.register_CFunction_SEOBNRv5_aligned_spin_unwrap()
BHaH.seobnr.utils.root_finding_1d.register_CFunction_root_finding_1d()
BHaH.seobnr.utils.root_finding_multidimensional.register_CFunction_root_finding_multidimensional()

# register SEOBNRv5 coefficients
BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients.register_CFunction_SEOBNRv5_aligned_spin_coefficients(
    False, False
)

# register initial condition routines
BHaH.seobnr.initial_conditions.SEOBNRv5_aligned_spin_multidimensional_root_wrapper.register_CFunction_SEOBNRv5_multidimensional_root_wrapper()
BHaH.seobnr.initial_conditions.SEOBNRv5_aligned_spin_Hamiltonian_circular_orbit.register_CFunction_SEOBNRv5_aligned_spin_Hamiltonian_circular_orbit()
BHaH.seobnr.initial_conditions.SEOBNRv5_aligned_spin_initial_conditions_conservative_nodf.register_CFunction_SEOBNRv5_aligned_spin_initial_conditions_conservative_nodf()
BHaH.seobnr.initial_conditions.SEOBNRv5_aligned_spin_radial_momentum_condition.register_CFunction_SEOBNRv5_aligned_spin_radial_momentum_condition()
BHaH.seobnr.initial_conditions.SEOBNRv5_aligned_spin_initial_conditions_dissipative.register_CFunction_SEOBNRv5_aligned_spin_initial_conditions_dissipative()

# register trajectory integration and processing routines
BHaH.seobnr.dynamics.eval_abs_deriv.register_CFunction_eval_abs_deriv()
BHaH.seobnr.dynamics.find_local_minimum_index.register_CFunction_find_local_minimum_index()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_augments.register_CFunction_SEOBNRv5_aligned_spin_augments()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_interpolate_dynamics.register_CFunction_SEOBNRv5_aligned_spin_interpolate_dynamics()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_iterative_refinement.register_CFunction_SEOBNRv5_aligned_spin_iterative_refinement()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_right_hand_sides.register_CFunction_SEOBNRv5_aligned_spin_right_hand_sides()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_ode_integration.register_CFunction_SEOBNRv5_aligned_spin_ode_integration()
BHaH.seobnr.dynamics.SEOBNRv5_aligned_spin_flux.register_CFunction_SEOBNRv5_aligned_spin_flux()

# register inspiral higher-mode waveform routines (example-specific, 9 modes)
BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_gamma_wrapper.register_CFunction_SEOBNRv5_aligned_spin_gamma_wrapper()
BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_waveform_9modes.register_CFunction_SEOBNRv5_aligned_spin_waveform()
BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_waveform_from_dynamics_9modes.register_CFunction_SEOBNRv5_aligned_spin_waveform_from_dynamics()
BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_interpolate_modes_9modes.register_CFunction_SEOBNRv5_aligned_spin_interpolate_modes()

pcg.do_parallel_codegen()
#########################################################
# STEP 3: Generate header files, register C functions and
#         command line parameters, and create a Makefile for
#         this project. Project is output to project/[project_name]/
BHaH.CodeParameters.write_CodeParameters_h_files(
    set_commondata_only=True, project_dir=project_dir
)
BHaH.CodeParameters.register_CFunctions_params_commondata_struct_set_to_default()
BHaH.cmdline_input_and_parfiles.generate_default_parfile(
    project_dir=project_dir, project_name=project_name
)
BHaH.cmdline_input_and_parfiles.register_CFunction_cmdline_input_and_parfile_parser(
    project_name=project_name,
    cmdline_inputs=["mass_ratio", "chi1", "chi2", "initial_omega", "total_mass", "dt"],
)

# Build the SEOBNR supplemental defines: NUMMODES = 1 (time) + number of output modes.
strain_defines = "\n".join(
    f"#define STRAIN{l}{m} {idx + 1}" for idx, (l, m) in enumerate(MODES)
)
seobnr_defines = f"""
#include<complex.h>
#define COMPLEX double complex
#define NUMVARS 8
#define TIME 0
#define R 1
#define PHI 2
#define PRSTAR 3
#define PPHI 4
#define H 5
#define OMEGA 6
#define OMEGA_CIRC 7
#define IDX(idx, var) ((idx)*NUMVARS + (var))
#define NUMMODES {len(MODES) + 1}
{strain_defines}
#define STRAIN 1
#define IDX_WF(idx,var) ((idx)*NUMMODES + (var))
typedef struct {{
  gsl_spline *spline;
  gsl_interp_accel *acc;
}} spline_data;
"""

additional_includes = [
    str(Path("gsl") / Path("gsl_vector.h")),
    str(Path("gsl") / Path("gsl_multiroots.h")),
    str(Path("gsl") / Path("gsl_errno.h")),
    str(Path("gsl") / Path("gsl_roots.h")),
    str(Path("gsl") / Path("gsl_matrix.h")),
    str(Path("gsl") / Path("gsl_odeiv2.h")),
    str(Path("gsl") / Path("gsl_spline.h")),
    str(Path("gsl") / Path("gsl_interp.h")),
    str(Path("gsl") / Path("gsl_sf_gamma.h")),
    str(Path("gsl") / Path("gsl_linalg.h")),
    "complex.h",
]
BHaH.BHaH_defines_h.output_BHaH_defines_h(
    project_dir=project_dir,
    additional_includes=additional_includes,
    enable_rfm_precompute=False,
    supplemental_defines_dict={"SEOBNR": seobnr_defines},
)
register_CFunction_main_c(
    output_waveform_flag,
    output_commondata_flag,
)

addl_cflags = ["$(shell gsl-config --cflags)"]
BHaH.Makefile_helpers.output_CFunctions_function_prototypes_and_construct_Makefile(
    project_dir=project_dir,
    project_name=project_name,
    exec_or_library_name=project_name,
    addl_CFLAGS=addl_cflags,
    addl_libraries=["$(shell gsl-config --libs)"],
)

print(
    f"Finished! Now go into project/{project_name} and type `make` to build, then ./{project_name} to run."
)
print(f"    Parameter file can be found in {project_name}.par")
