# NRPy Agent Instructions

This file translates [coding_style.md](/home/zetienne/virt/nrpy/coding_style.md) into agent-facing instructions. Follow these rules for all edits unless the user explicitly asks otherwise. CI enforces many of them.

## Scope

- NRPy generates C code from Python/SymPy expressions.
- Treat `coding_style.md` as source of truth if this file and that file ever diverge.
- Binary files are not allowed in pull requests. Do not add images, archives, compiled artifacts, or other non-text assets unless maintainers explicitly approve an exception.

## Required Checks

- Run `black .` before committing Python changes.
- Run `./.github/single_file_static_analysis.sh <path-to-file.py>` on every modified Python file before committing.
- For Python changes, do not run static analysis on only a subset of touched files.
- All contributions must pass project static analysis before merge.

Static analysis script enforces:

- `black --check`
- `isort --check-only`
- `mypy --strict --allow-untyped-calls`
- `pylint` with project config and threshold `>= 9.91/10`
- `pydocstyle`
- `darglint -v 2`
- doctests by executing `python3 <file>`

## Python Style

### Formatting

- Follow Black output for line wrapping and formatting.
- Use standard 4-space Python indentation.

### Naming

- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` or leading underscore for internal constants
- Private helpers: leading underscore
- Boolean or boolean-like parameters must use positive names

Prefer:

- `enable_feature`
- `include_header`
- `allow_resize`
- `has_ghost_zones`

Avoid:

- `disable_feature`
- `omit_header`
- `forbid_resize`
- `no_ghost_zones`

### Imports

- Order imports as:
  1. Standard library
  2. Third-party
  3. Local NRPy modules
- Separate each group with one blank line.
- Follow `isort`.

Canonical aliases are mandatory:

- `import sympy as sp`
- `import nrpy.indexedexp as ixp`
- `import nrpy.params as par`
- `import nrpy.grid as gri`
- `import nrpy.reference_metric as refmetric`
- `import nrpy.c_function as cfc`
- `import nrpy.c_codegen as ccg`
- `import nrpy.helpers.parallel_codegen as pcg`

Do not invent alternate aliases.

### `__init__.py`

- No module docstring.
- No comments.
- No executable code beyond explicit relative import aggregation.
- Keep namespace flat with explicit relative imports.

### Python Docstrings

- Use triple double quotes: `"""`
- Use Sphinx/reST style only
- Use `:param name:`
- Use `:return:` not `:returns:`
- Use `:raises ExceptionType:`
- Never use Google-style `Args:`, `Returns:`, `Raises:`
- Do not put type information into `:param` descriptions

### Module Docstrings

For every non-`__init__.py` Python file, put a module docstring at top with this structure:

```python
"""
<Description paragraphs.>

Author: Name
        email obfuscated
"""
```

For multiple authors:

```python
"""
<Description paragraphs.>

Authors: Name One
         email
         Name Two
         email
"""
```

Rules:

- Use `Author:` for exactly one author
- Use `Authors:` for two or more
- Do not add authors who are not already authors of file
- Email obfuscation encouraged, exact format not enforced
- Optional leading filename comment allowed immediately before module docstring only if exact form is `# <relative path from nrpy root>.py`
- `__init__.py` files never have module docstrings
- Trusted test-vector files never have module docstrings

Avoid these anti-patterns:

- `Author:` with multiple names
- `Email:` or `Contributor:` metadata keys
- Mixed metadata key styles in one file

### Type Hints

- Use type hints extensively.
- Always include return annotations, including `-> None`.
- Use `typing` forms such as `Dict`, `List`, `Optional`, `Tuple`, `Union`, `cast`.
- Use `typing_extensions.Literal` for constrained strings.
- Use `# type: ignore` only selectively for third-party gaps.
- Do not use `Any` when a more precise type is possible.
- If `Any` is unavoidable, document reason inline.
- Do not use Python 3.9+ builtin generics like `list[X]`, `dict[X, Y]`, `tuple[X, ...]`.
- Do not use `X | None` union shorthand.
- Use `List[X]`, `Dict[X, Y]`, `Optional[X]`, `Union[X, Y]`.
- Do not add `from __future__ import annotations` to files that do not already use it unless there is a specific need such as forward references in same file.

### Comments

Inside docstrings:

- Inline `Note:` prose is acceptable
- reST `.. note::` blocks are also acceptable
- Do not mix both styles inside one docstring

Procedural code comment structure:

- Use `# Step N:`
- Substeps use `# Step 1.a:`
- Module-level preamble steps use `# Step P1:`
- Use lowercase `Step`, not uppercase `STEP`

### `if __name__ == "__main__":`

Every runnable non-test, non-`__init__.py` file must end with this doctest runner pattern:

```python
if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
```

- Imports used only by `__main__` block belong inside block, not at module top.

### General Python Organization

- Classes group related functionality.
- Module-level functions handle procedural code generation.
- Registration pattern via `register_CFunction_*()` is common.
- Doctests live in docstrings.
- Raise `ValueError` for invalid inputs.
- Use `warnings.warn()` for non-critical issues.
- Validate explicitly in constructors.

## Equation Setup Rules

### SymPy

- Avoid `sp.simplify()` in equation-building code.
- `sp.simplify()` allowed only in test/validation code or explicit identity checks.
- Do not use `sp.subs()` or `sp.replace()` for pattern-based expression transformation.

Allowed `.subs()` cases:

- Coordinate substitution in elliptic source terms
- Face-value substitution in GRHD fluxes
- Systematic `nrpyABS` to `sp.Abs` conversion
- Surface radius substitution in horizon modules
- Evaluating expression at specific parameter value such as `.subs(t, t_attach)`

Preferred patterns:

- Initialize accumulators with `sp.sympify(0)` and `sp.sympify(1)`
- Use `sp.Rational()` for exact fractions
- Use `sp.symbols(..., real=True)` for scalar quantities

### Indexed Expressions

Use `nrpy.indexedexp` imported as `ixp` as primary tensor interface.

Common helpers:

- `ixp.zerorank1()` through `ixp.zerorank4()`
- `ixp.declarerank1()` through `ixp.declarerank4()`
- `ixp.symm_matrix_inverter2x2/3x3/4x4()`
- `ixp.LeviCivitaSymbol_dim3_rank3()`
- `ixp.LeviCivitaTensorUUU_dim3_rank3()`

Symmetry strings:

- `"sym01"` for metrics
- `"sym12"` for derivatives of symmetric tensors
- `"sym01_sym23"` for Riemann-like tensors
- `"nosym"` for non-symmetric arrays

### Expression Construction

- Use explicit nested loops for tensor accumulation.
- No Einstein summation notation.
- No matrix multiplication operators for this work pattern.

### Symbol Naming

Standard suffixes:

- `U`: contravariant
- `D`: covariant
- `DD` and `UU`: rank-2 covariant or contravariant
- `dD` and `dDD`: first and second partial derivatives
- `dupD`: upwinded derivative
- `dBarD` and `dHatD`: conformal or reference-metric covariant derivative
- `rhs`: evolution RHS

Derivative naming rules:

- First partial derivatives use `*_dD`
- Second partial derivatives use `*_dDD`
- Finite-difference-like derivative names must include `dD` or `dDD`
- Upwinded derivatives use `dupD`
- Declare derivatives with `ixp` helpers so generated names and suffixes stay consistent

### Expression Validation

Every equation module validates symbolic expressions against trusted numerical values.

Standard pipeline:

1. Build `results_dict` mapping expression names to SymPy expressions.
2. Convert expressions numerically with `ve.process_dictionary_of_expressions(...)`.
3. Compare or generate trusted results with `ve.compare_or_generate_trusted_results(...)`.

Use `nrpy.validate_expressions.validate_expressions as ve`.

Key APIs:

- `assert_equal(vardict_1, vardict_2)`
- `check_zero(expression)`
- `process_dictionary_of_expressions(dict)`
- `compare_against_trusted(...)`
- `output_trusted(...)`
- `compare_or_generate_trusted_results(...)`

Trusted file rules:

- Trusted vectors live under `*/tests/`
- No module docstrings
- No functions or classes
- Only `mpf` or `mpc` imports, with `# type: ignore` as used elsewhere
- Do not hand-edit trusted values
- Regenerate them from owning module
- If mismatch is legitimate, delete stale trusted file and rerun module
- Commit message must explain why trusted file changed

Equation output contract:

- Assign key outputs to `self.<expr_name>`
- Validate with same `process_dictionary_of_expressions` and `compare_or_generate_trusted_results` pattern
- Result names must match corresponding `trusted_dict` keys
- Use existing naming conventions such as `*_rhs` and `*_expr_list_*`
- Do not invent new naming conventions without aligning trusted files

### Prohibited or Restricted Dependencies

- Do not depend on `numpy`
- NRPy workflow is symbolic SymPy to generated C

Regex rule:

- Do not `import re` when plain `.replace()` or similar string methods are enough
- Use regex only for genuine pattern matching that plain string logic cannot robustly handle
- If regex is justified, add comment explaining why `.replace()` is insufficient

## Infrastructure Code Rules

### Module Organization

- Each Python infrastructure file usually registers one primary C function via `register_CFunction_*()`
- Python module names use descriptive `snake_case`
- `__init__.py` stays flat and uses explicit relative imports
- Standard structure:
  1. Module docstring with author info
  2. Imports
  3. Main `register_CFunction_*()` function
  4. Optional helper functions only when genuinely useful
  5. Doctest `__main__` block

### Doctests

Registration function docstrings should end with `Doctests:` immediately before first `>>>`.

Use `Doctests:` in new code, not:

- `Doctest:`
- `DocTests:`

Preferred doctest pattern for stable emitted C:

- Import `validate_strings` and `clang_format` inside doctest
- Clear `cfc.CFunction_dict` before registration to avoid test pollution
- Normalize `full_function` with `clang_format`
- Compare with `validate_strings`
- Use `file_ext="c"` normally
- Use `file_ext="cu"` for CUDA

Template:

```python
Doctests:
>>> from nrpy.helpers.generic import validate_strings, clang_format
>>> import nrpy.c_function as cfc
>>> import nrpy.params as par
>>> cfc.CFunction_dict.clear()
>>> _ = register_CFunction_foo("Spherical")
>>> generated_str = clang_format(cfc.CFunction_dict["foo__rfm__Spherical"].full_function)
>>> _ = validate_strings(generated_str, "foo__openmp__Spherical", file_ext="c")
```

Do not use golden-output doctests for generated-kernel-heavy C functions whose output changes too easily with SymPy/codegen details. Prefer symbolic validation or lighter structural checks.

If you see `# FIXME` doctest placeholders:

- Treat as temporary scaffolding
- Keep new doctests lightweight
- Complete or remove placeholders when ready

Do not write doctests whose main value is only proving `register_CFunction_*()` ran or inserted into `cfc.CFunction_dict`.

### Parallel Codegen Pattern

If registration uses `nrpy.helpers.parallel_codegen`, implement discovery-phase early return:

```python
def register_CFunction_my_function() -> Union[None, pcg.NRPyEnv_type]:
    """Register my C function."""
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None
    # actual registration logic
    return pcg.NRPyEnv()
```

Rules:

- Apply this only to registration functions that participate in parallel/discovery codegen
- Register C function only after guard
- Functions not using parallel codegen should return `-> None`

### Black Suppression

- Use `# fmt: off` / `# fmt: on` sparingly
- Only when Black would destroy intentional alignment
- Common case: aligned `par.CodeParameter(...)` lines

### C Function Registration from Python

Use `cfc.register_CFunction()` with standard named parameters such as:

- `subdirectory`
- `includes`
- `prefunc`
- `desc`
- `cfunc_type`
- `name`
- `params`
- `include_CodeParameters_h`
- `body`
- `postfunc`

Registration-function structure:

- Declare `desc`, `cfunc_type`, `name`, `params`, `body`, and similar values on separate lines
- Keep function self-contained and readable in execution order
- Do not split one-off setup into tiny helpers unless logic is genuinely reusable

### Embedded C in Python Strings

- Use raw strings `r"""..."""` for C bodies
- Use `rf"""..."""` when interpolation needed
- Double C braces inside f-strings: `{{` and `}}`
- Embedded C indentation uses 2 spaces, regardless of Python indentation level
- Minor historical indentation variance may remain; do not churn valid code purely for that
- If `include_CodeParameters_h=True`, embedded bodies use `#include "set_CodeParameters.h"`
- String replacement can adapt generated C when needed, for example replacing array prefixes

### BHaH Symbolic Codegen Rules

For new ordinary per-grid or per-point kernel generators:

- Prefer `BHaH.simple_loop.simple_loop()` over handwritten `LOOP_OMP(...)` or raw nested loops
- Prefer `ccg.c_codegen(..., automatically_read_gf_data_from_memory=True)` for registered gridfunction reads instead of manual repetitive loads
- Keep transformations symbolic until `c_codegen()` whenever practical
- Do not use string-based replacement of symbolic expressions or generated C when symbolic expression can express same result
- Do not add new `#include "set_CodeParameters.h"` lines inside generated function bodies for this class of infrastructure code
- Instead derive needed `params` and `commondata` locals with `get_params_commondata_symbols_from_expr_list()` and `generate_definition_header()`
- Avoid post-registration mutation of `cfc.CFunction_dict[...]` bodies as customization mechanism
- Prefer explicit extension hooks or helper parameters in shared registration logic
- Keep registration functions linear
- Do not hide one-off symbolic setup in private helpers that force scrolling
- Construct symbolic expressions immediately before consuming them in `c_codegen()` or `simple_loop()`
- Register gridfunctions, parity tables, and similar metadata near point of actual need
- Do not front-load unrelated setup before basic metadata such as `desc`, `name`, `params`
- When assembling emitted C bodies, prefer top-to-bottom `body += ...` that mirrors C execution order
- Add short comments at jarring transitions so readers understand abrupt flow changes

### Inlining Rules

Inline all C and Python functions that do not save lines by existing separately.

Separate a function only if:

- It is called from multiple locations
- Or it contains more than roughly 10 to 15 lines of actual logic

Avoid:

- Single-use 2 to 5 line helpers
- Functions whose docstring is longer than body
- Functions that merely return one expression

Inline simple expressions directly where used.

### `prefunc`

- Helper C functions are emitted as strings into `prefunc`
- Concatenate helpers with `prefunc += ...`

### Standard Struct Pointer Parameters

Common C parameter patterns:

- `commondata_struct *restrict commondata`
- `griddata_struct *restrict griddata`
- `params_struct *restrict params`
- `bc_struct *restrict bcstruct`

### Gridfunction Naming and Grouping

Naming patterns:

- Scalars like `HHGF`, `VVGF`, `WWGF`, `TRKGF`, `CFGF`
- Symmetric rank-2 tensors like `HDD00GF`, `HDD01GF`, `HDD11GF`
- Traceless extrinsic curvature like `ADD00GF`
- Derivatives like `SRC_PARTIAL_D_HDD000GF`

Groups:

- `EVOL`: time-evolved quantities
- `AUXEVOL`: auxiliary evolution quantities
- `AUX`: diagnostics

### Indexing Macros

Common macros:

- `IDX3(i0, i1, i2)`
- `IDX4(gf, i0, i1, i2)`
- `IDX4pt(gf, idx3)`
- `IDX2(i1, i2)`

Custom variants such as `SRC_IDX4`, `DST_IDX4`, `EX_IDX4` are allowed where appropriate.

### Performance and Parallelism Patterns

SIMD:

- Use `sum_lagrange_x0_simd()` from `interpolation_lagrange_uniform.h` for vectorized inner loops
- Use `#pragma omp simd` for GF-level vectorization
- `simd_intrinsics.h` may be copied into project when SIMD disabled elsewhere

OpenMP:

- `#pragma omp parallel for`
- `#pragma omp parallel for reduction(+:var)`
- `#pragma omp parallel for collapse(2)`
- `#pragma omp critical`
- `LOOP_OMP("omp parallel for", ...)` for multidimensional loops

### Memory and Error Handling

Memory:

- Prefer `BHAH_MALLOC` and `BHAH_FREE` for tracked allocations
- Standard `malloc()` and `free()` allowed for simple cases
- Check all allocations for `NULL`
- Prefer one large heap allocation over many small ones
- VLAs are used for stack temporary buffers

Errors:

- Use enum-based error codes
- Check returned error codes immediately
- Common pattern: `if (commondata->error_flag != BHAHAHA_SUCCESS) return;`
- In parallel regions, use `#pragma omp critical` when setting shared error flags

### SymPy to C

- Use `ccg.c_codegen()` to generate C from SymPy expressions
- Often called with lists of expressions and target names
- Derivatives typically declared via `ixp.declarerankN()` before codegen

### Preprocessor and Comment Patterns

Preprocessor:

- Use `#ifndef REAL` / `#define REAL double` / `#endif` style for portability
- Use `#ifndef M_PI` fallback if needed
- Use `#ifdef STANDALONE` for standalone tests
- Use GCC optimization pragmas only where appropriate
- Use `MAYBE_UNUSED` for intentionally unused variables

C comments:

- In function bodies and general C code, use `//`, not block comments
- Function documentation comes from Doxygen comments or `desc=`
- Struct fields are grouped with clear `//==========================` separators

## C/H Style

### Formatting

- Use 2 spaces
- No tabs
- Aim for roughly 100-character lines
- Use K&R braces for functions and control flow

### Naming

- Structs: `snake_case` with `_struct` suffix when using full struct types
- Functions: `snake_case`
- Macros: `UPPER_SNAKE_CASE`
- Variables: `snake_case`
- Enums: `UPPER_SNAKE_CASE`

### Header Guards and Includes

- Use traditional `#ifndef` / `#define` / `#endif` guards
- Guard names are `UPPER_SNAKE_CASE`
- Both simple and double-underscore styles exist; follow surrounding file style
- Put standard library includes first
- Put project headers in quotes
- Use conditional includes for platform-specific headers

### Macros and Declarations

- Wrap macro arguments in parentheses
- Use `#if defined(...)`, `#elif defined(...)`, `#endif`
- Header-defined helpers should be `static inline`
- Keep return type on same line as function name
- Use `const` and `restrict` qualifiers where appropriate
- Declare variables at point of first use

### End-Curly-Brace Comments

Every closing brace ending a non-trivial block in new C code must carry a `// END ...` comment unless block body is fewer than 5 lines and opening brace is still visible without scrolling.

Formats:

- Function: `} // END FUNCTION: name`
- `for`: `} // END LOOP: for <var> over <range/purpose>`
- `while`: `} // END WHILE: brief description`
- `if`: `} // END IF: brief condition`
- `else if`: `} // END ELSE IF: brief condition`
- `else`: `} // END ELSE: brief description`
- OpenMP parallel: `} // END OMP PARALLEL`
- OpenMP parallel for: `} // END OMP PARALLEL FOR`
- Anonymous block: `} // END BLOCK: description`
- `do ... while`: `} while (condition); // END DO-WHILE: brief description`

Rules:

- Keyword after `// ` is all caps
- Use colon except OpenMP forms that mirror pragma syntax
- No trailing period
- No parentheses after function names

### Doxygen for C

Every C function with body longer than about 10 lines needs Doxygen immediately above declaration or definition.

Placement:

- If function is declared in header and defined in `.c`, put comment on declaration
- Otherwise put comment on definition

Structure:

- Opening `/**` on own line
- Closing `*/` on own line
- First content line is brief description, no `@brief`
- Always include blank ` *` line after brief
- If extended description exists, add another blank ` *` before tags
- For complex functions, prefer numbered step descriptions over dense prose
- Tag order: `@param` block, then `@return`, then blank ` *`, then `@note` / `@warning` / `@pre`
- Do not embed `@warning` inside `@param`
- Keep `@param` descriptions to one line
- No dash after parameter name
- Use `@param[in]` for `const` pointers
- Use `@param[out]` or `@param[in,out]` for mutable pointers as appropriate
- Use plain `@param` for pass-by-value parameters
- Omit `@return` for `void`
- For integer error-code returns, enumerate outcomes or mention enum values

The same tag conventions apply to `desc=` strings passed into `register_CFunction()`, except `/**` and `*/` must not appear there because framework emits them.

## Additional Project Rules

- When conditional Python logic builds C body text, incremental `body += ...` construction is accepted and preferred over forcing all cases into one raw string.
- Use this pattern especially when optional sections depend on Python flags.

## Quick Reference

Python:

- 4 spaces
- Black + isort
- Sphinx docstrings
- Extensive old-style `typing` annotations
- No docstrings in `__init__.py`
- Doctest runner block at end of runnable files

C:

- 2 spaces
- K&R braces
- Doxygen comments
- `// END ...` comments for non-trivial closing braces
- Header guards with `#ifndef`

Testing and validation:

- Static-analysis script on every modified Python file
- Trusted vectors under `tests/`
- Regenerate trusted outputs, do not hand-edit
- Favor symbolic validation over brittle golden generated-C output for large generated kernels
