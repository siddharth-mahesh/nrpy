"""
Set up C function library for interpolating the SEOBNRv5 aligned-spin inspiral modes (9 modes).

Example-specific variant of ``SEOBNRv5_aligned_spin_interpolate_modes`` (which handles only the
(2,2) mode). This version concatenates the low- and fine-sampled inspiral modes stored in
``waveform_low``/``waveform_fine`` and resamples all 9 delta-implemented modes (excluding (8,8))
onto a uniform time grid, storing the result in ``waveform_inspiral``.

Each mode ``(l,m)`` is de-phased by ``exp(+i m phi)`` (using the orbital phase) before cubic-spline
interpolation and re-phased by ``exp(-i m phi)`` afterwards, generalizing the (2,2)-only approach.

Authors: Siddharth Mahesh
        sm0193 **at** mix **dot** wvu **dot** edu
        Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
from nrpy.infrastructures.BHaH.seobnr.inspiral_waveform.SEOBNRv5_aligned_spin_waveform_9modes import (
    MODES,
)


def register_CFunction_SEOBNRv5_aligned_spin_interpolate_modes() -> (
    Union[None, pcg.NRPyEnv_type]
):
    """
    Register CFunction for interpolating all 9 SEOBNRv5 inspiral modes onto a uniform grid.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = """
Concatenates the low- and fine-sampled inspiral modes and interpolates all 9 modes
(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (5,5), (6,6), (7,7) onto a uniform time grid,
storing them in the waveform_inspiral array.

@param commondata - Common data structure containing the model parameters.
@param dT - Time step for interpolation.
"""
    cfunc_type = "void"
    name = "SEOBNRv5_aligned_spin_interpolate_modes"
    params = "commondata_struct *restrict commondata, const REAL dT"

    mode_blocks = ""
    for l, m in MODES:
        mode_blocks += f"""
// mode ({l},{m})
for (i = 0; i < commondata->nsteps_low; i++){{
  mode_nophase = cexp({m} * I * orbital_phases[i]) * (commondata->waveform_low[IDX_WF(i,STRAIN{l}{m})]);
  mode_real[i] = creal(mode_nophase);
  mode_imag[i] = cimag(mode_nophase);
}}
for (i = 0; i < commondata->nsteps_fine; i++){{
  mode_nophase = cexp({m} * I * orbital_phases[i + commondata->nsteps_low]) * (commondata->waveform_fine[IDX_WF(i,STRAIN{l}{m})]);
  mode_real[i + commondata->nsteps_low] = creal(mode_nophase);
  mode_imag[i + commondata->nsteps_low] = cimag(mode_nophase);
}}
gsl_spline_init(spline_real, times_old, mode_real, nsteps_old);
gsl_spline_init(spline_imag, times_old, mode_imag, nsteps_old);
gsl_interp_accel_reset(acc_real);
gsl_interp_accel_reset(acc_imag);
for (i = 0; i < commondata->nsteps_inspiral; i++){{
  time = tstart + i * dT;
  mode_real_interp = gsl_spline_eval(spline_real, time, acc_real);
  mode_imag_interp = gsl_spline_eval(spline_imag, time, acc_imag);
  mode_rescaled = cexp(-{m} * I * orbital_phase_new[i]) * (mode_real_interp + I * mode_imag_interp);
  commondata->waveform_inspiral[IDX_WF(i,STRAIN{l}{m})] = mode_rescaled;
}}
"""

    body = (
        """
size_t i;
const size_t nsteps_old = commondata->nsteps_low + commondata->nsteps_fine;

// Assemble the (non-uniform) time axis and orbital phase from the dynamics.
REAL *restrict times_old = (REAL *)malloc(nsteps_old * sizeof(REAL));
if (times_old == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for times_old\\n");
  exit(1);
}
REAL *restrict orbital_phases = (REAL *)malloc(nsteps_old * sizeof(REAL));
if (orbital_phases == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for orbital_phases\\n");
  exit(1);
}
for (i = 0; i < commondata->nsteps_low; i++){
  times_old[i] = commondata->dynamics_low[IDX(i,TIME)];
  orbital_phases[i] = commondata->dynamics_low[IDX(i,PHI)];
}
for (i = 0; i < commondata->nsteps_fine; i++){
  times_old[i + commondata->nsteps_low] = commondata->dynamics_fine[IDX(i,TIME)];
  orbital_phases[i + commondata->nsteps_low] = commondata->dynamics_fine[IDX(i,PHI)];
}

const REAL tstart = times_old[0];
const REAL tend = times_old[nsteps_old - 1];
commondata->nsteps_inspiral = (size_t) ((tend - tstart) / dT) + 1;

// Spline the orbital phase and pre-evaluate it on the uniform grid.
gsl_interp_accel *restrict acc = gsl_interp_accel_alloc();
if (acc == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_interp_accel_alloc failed to initialize\\n");
  exit(1);
}
gsl_spline *restrict spline = gsl_spline_alloc(gsl_interp_cspline, nsteps_old);
if (spline == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_spline_alloc failed to initialize\\n");
  exit(1);
}
gsl_spline_init(spline, times_old, orbital_phases, nsteps_old);

REAL *restrict orbital_phase_new = (REAL *)malloc(commondata->nsteps_inspiral * sizeof(REAL));
if (orbital_phase_new == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for orbital_phase_new\\n");
  exit(1);
}
for (i = 0; i < commondata->nsteps_inspiral; i++){
  orbital_phase_new[i] = gsl_spline_eval(spline, tstart + i * dT, acc);
}

// Allocate the uniformly-sampled inspiral waveform and set the time column.
commondata->waveform_inspiral = (double complex *)malloc(commondata->nsteps_inspiral * NUMMODES * sizeof(double complex));
if (commondata->waveform_inspiral == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for waveform_inspiral\\n");
  exit(1);
}
for (i = 0; i < commondata->nsteps_inspiral; i++){
  commondata->waveform_inspiral[IDX_WF(i,TIME)] = tstart + i * dT;
}

// Per-mode scratch buffers and splines (reused across modes).
REAL *restrict mode_real = (REAL *)malloc(nsteps_old * sizeof(REAL));
if (mode_real == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for mode_real\\n");
  exit(1);
}
REAL *restrict mode_imag = (REAL *)malloc(nsteps_old * sizeof(REAL));
if (mode_imag == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), malloc() failed to for mode_imag\\n");
  exit(1);
}
gsl_interp_accel *restrict acc_real = gsl_interp_accel_alloc();
if (acc_real == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_interp_accel_alloc failed to initialize\\n");
  exit(1);
}
gsl_interp_accel *restrict acc_imag = gsl_interp_accel_alloc();
if (acc_imag == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_interp_accel_alloc failed to initialize\\n");
  exit(1);
}
gsl_spline *restrict spline_real = gsl_spline_alloc(gsl_interp_cspline, nsteps_old);
if (spline_real == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_spline_alloc failed to initialize\\n");
  exit(1);
}
gsl_spline *restrict spline_imag = gsl_spline_alloc(gsl_interp_cspline, nsteps_old);
if (spline_imag == NULL){
  fprintf(stderr,"Error: in SEOBNRv5_aligned_spin_interpolate_modes(), gsl_spline_alloc failed to initialize\\n");
  exit(1);
}
double complex mode_nophase, mode_rescaled;
REAL time, mode_real_interp, mode_imag_interp;
"""
        + mode_blocks
        + """
gsl_interp_accel_free(acc);
gsl_interp_accel_free(acc_real);
gsl_interp_accel_free(acc_imag);
gsl_spline_free(spline);
gsl_spline_free(spline_real);
gsl_spline_free(spline_imag);
free(times_old);
free(orbital_phases);
free(orbital_phase_new);
free(mode_real);
free(mode_imag);
"""
    )
    cfc.register_CFunction(
        subdirectory="inspiral_waveform",
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return pcg.NRPyEnv()
