"""
Module for constructing interface.ccl for Cactus thorns.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
        Samuel Cupp
"""

from typing import List
from pathlib import Path
import nrpy.grid as gri
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater


def construct_interface_ccl(
    project_dir: str,
    thorn_name: str,
    inherits: str,
    USES_INCLUDEs: str,
    is_evol_thorn: bool = False,
    enable_NewRad: bool = False,
) -> None:
    """
    Generate `interface.ccl` file required for the specified Thorn.

    :param thorn_name: The name of the thorn for which the interface.ccl file is generated. Defaults to "BaikalETK".
    :param enable_stress_energy_source_terms: Boolean flag to determine whether to include stress-energy source terms. Defaults to False.
    :return: None
    """
    outstr = rf"""
# This interface.ccl file was automatically generated by NRPy+.
#   You are advised against modifying it directly; instead
#   modify the Python code that generates it.

# With "implements", we give our thorn its unique name.
implements: {thorn_name}

# By "inheriting" other thorns, we tell the Toolkit that we
#   will rely on variables/function that exist within those
#   functions.
inherits: {inherits}

# Needed functions and #include's:
{USES_INCLUDEs}
"""
    if enable_NewRad:
        outstr += r"""
# Note: we don't have NewRad yet
# Needed for NewRad outer boundary condition driver:
#CCTK_INT FUNCTION                         \
#    NewRad_Apply                          \
#        (CCTK_POINTER_TO_CONST IN cctkGH, \
#         CCTK_REAL ARRAY IN var,          \
#         CCTK_REAL ARRAY INOUT rhs,       \
#         CCTK_REAL IN var0,               \
#         CCTK_REAL IN v0,                 \
#         CCTK_INT IN radpower)
#REQUIRES FUNCTION NewRad_Apply
"""
    outstr += """
# Tell the Toolkit that we want all gridfunctions
#    to be visible to other thorns by using
#    the keyword "public". Note that declaring these
#    gridfunctions *does not* allocate memory for them;
#    that is done by the schedule.ccl file.

public:
"""
    (
        evolved_variables_list,
        auxiliary_variables_list,
        auxevol_variables_list,
    ) = gri.CarpetXGridFunction.gridfunction_lists()[0:3]

    def construct_parity_string(parities: List[int]) -> str:
        """
        Construct the parities tag for a variable group using the list of parity values
        given by set_parity_types

        :param list_of_gf_names: List of grid function names for which to set the parity types.
        :return: A list of integers representing the parity types for the grid functions.
        """
        parity_tag = ""
        for parity_value in parities:
            parity_tag += "  "
            # scalar and tensor xx, yy, zz components
            if parity_value in [0, 4, 7, 9]:
                parity_tag += "+1 +1 +1"
            # vector components
            elif parity_value == 1:
                parity_tag += "-1 +1 +1"
            elif parity_value == 2:
                parity_tag += "+1 -1 +1"
            elif parity_value == 3:
                parity_tag += "+1 +1 -1"
            # tensor off-diagonal components
            elif parity_value == 5:  # xy
                parity_tag += "-1 -1 +1"
            elif parity_value == 6:  # xz
                parity_tag += "-1 +1 -1"
            elif parity_value == 8:  # yz
                parity_tag += "+1 -1 -1"

        return parity_tag

    if is_evol_thorn:
        if evolved_variables_list:
            # First, EVOL type:
            evol_gfs = ", ".join([evol_gf + "GF" for evol_gf in evolved_variables_list])
            evol_parity_type = gri.CarpetXGridFunction.set_parity_types(
                evolved_variables_list
            )
            evol_parities = construct_parity_string(evol_parity_type)
            outstr += f"""CCTK_REAL evol_variables type = GF Timelevels=1 TAGS='rhs="evol_variables_rhs" parities={{{evol_parities}}}'
{{
  {evol_gfs}
}} "Evolved gridfunctions."

"""

            # Second EVOL right-hand-sides
            rhs_gfs = ", ".join(
                [evol_gf + "_rhsGF" for evol_gf in evolved_variables_list]
            )
            outstr += f"""CCTK_REAL evol_variables_rhs type = GF Timelevels=1 TAGS='InterpNumTimelevels=1 prolongation="none" checkpoint="no"'
{{
  {rhs_gfs}
}} "Right-hand-side gridfunctions."

"""

            # Then AUXEVOL type:
            if auxevol_variables_list:
                auxevol_gfs = ", ".join(
                    [auxevol_gf + "GF" for auxevol_gf in auxevol_variables_list]
                )
                auxevol_parity_type = gri.CarpetXGridFunction.set_parity_types(
                    auxevol_variables_list
                )
                auxevol_parities = construct_parity_string(auxevol_parity_type)
                outstr += f"""CCTK_REAL auxevol_variables type = GF Timelevels=1 TAGS='InterpNumTimelevels=1 prolongation="none" checkpoint="no" parities={{{auxevol_parities}}}'
{{
  {auxevol_gfs}
}} "Auxiliary gridfunctions needed for evaluating the RHSs."

"""

        # Then AUX type:
        if auxiliary_variables_list:
            aux_gfs = ", ".join([aux_gf + "GF" for aux_gf in auxiliary_variables_list])
            aux_parity_type = gri.CarpetXGridFunction.set_parity_types(
                auxiliary_variables_list
            )
            aux_parities = construct_parity_string(aux_parity_type)
            outstr += f"""CCTK_REAL aux_variables type = GF Timelevels=1 TAGS='parities={{{aux_parities}}}'
{{
  {aux_gfs}
}} "Auxiliary gridfunctions for e.g., diagnostics."

"""

    output_Path = Path(project_dir) / thorn_name
    output_Path.mkdir(parents=True, exist_ok=True)
    with ConditionalFileUpdater(
        output_Path / "interface.ccl", encoding="utf-8"
    ) as file:
        file.write(outstr)
