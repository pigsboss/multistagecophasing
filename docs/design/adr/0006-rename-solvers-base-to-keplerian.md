# ADR-0006: Rename solvers/base.py to solvers/keplerian.py

## Status

Accepted

## Context

The module `mission_sim/utils/solvers/base.py` contained specialised functions
for solving Kepler’s equation and converting orbital elements to Cartesian
coordinates.  Its generic name `base.py` was misleading; the file did not
serve as an abstract base class but instead provided concrete Keplerian‑orbit
utilities (elliptic, hyperbolic, parabolic batch converters and a unified
entry point).

Future additions such as Lambert solvers, three‑body correctors, or other
celestial‑mechanics tools would naturally reside under the `solvers` package.
A clear naming convention `solvers/<method>.py` makes the package self‑-
documenting and simplifies navigation for contributors.

## Decision

1. **Rename** `mission_sim/utils/solvers/base.py` to
   `mission_sim/utils/solvers/keplerian.py`.

2. **Update imports** in all files that referenced the old name:
   - `mission_sim/core/spacetime/ephemeris/kepler_ephemeris.py`
   - `mission_sim/core/spacetime/ephemeris/jpl_ssb_keplerian_elements.py`
   - `tests/test_analytical_ephemeris.py`

   The import statement
   ```python
   from mission_sim.utils.solvers.base import kepler_elements_to_cartesian_batch
   ```
   is changed to
   ```python
   from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch
   ```

3. No public API is modified; the function `kepler_elements_to_cartesian_batch`
   remains available under the same name and signature.

## Consequences

- **Positive:**
  - The file name now clearly communicates its purpose (Keplerian orbit
    solvers).
  - Opens a predictable naming convention for future solvers
    (`solvers/lambert.py`, `solvers/threebody.py`, etc.).
  - Reduces confusion when browsing the source tree or debugging import
    paths.

- **Negative:**
  - A one‑time search‑and‑replace must be performed across the codebase.
    A global `grep` confirmed that only three files required changes.

- **Risks:**
  - External code referencing `solvers.base` would break.  This is unlikely
    for an internal utility module, but a deprecation shim can be added if
    needed.

- **Transition plan:**
  - All imports were updated in the three affected files.
  - The existing test suite (`pytest tests/test_analytical_ephemeris.py`)
    passes after the rename, confirming no regressions.
  - The old `base.py` file was removed and the new `keplerian.py` is the
    only source of truth.

## References

- Renamed module: `mission_sim/utils/solvers/keplerian.py`
- Affected consumers: `kepler_ephemeris.py`, `jpl_ssb_keplerian_elements.py`,
  `test_analytical_ephemeris.py`
