# ADR-0005: Runge‑Kutta integrator family with factory pattern

## Status

Accepted

## Context

High‑order numerical propagation of dynamical systems requires a selection of
explicit Runge‑Kutta methods (e.g., RK45, DOP853, DP8(7)).  Hard‑coding each
method separately leads to duplicated step‑size control logic and Butcher table
storage.  Moreover, using a single generic function that reads the table at
run‑time prevents Numba from recognising the stage count as a compile‑time
constant, which degrades performance.

## Decision

We introduce a **factory pattern** that generates dedicated `@njit` stepper
functions for each Butcher tableau.  A shared adaptive‑loop harness removes the
remaining code duplication.

1. **Data container**  
   A `RKTable` named tuple stores the number of stages `s`, the arrays `C`, `A`,
   the higher‑order weight vector `B_high`, the lower‑order weight vector
   `B_low`, and the formal order `order`.

2. **Permanent module‑level tables**  
   - `TABLE_RK45`   – Dormand‑Prince 5(4), 7 stages  
   - `TABLE_DOP853` – Dormand‑Prince 8(5,3), 12 stages (coefficients extracted
     from `dop853.f`)  
   - `TABLE_DP8`    – placeholder for a 13‑stage 8(7) pair; currently aliased to
     `TABLE_DOP853` until verified coefficients become available.

3. **Stepper factory**  
   `_make_rk_step(table)` returns a `@njit`‑compiled function that unrolls the
   stage loop using `table.s` as a literal integer.  It computes both the
   high‑order and low‑order solutions and returns the stress‑scaled error norm.

4. **Shared adaptive loop**  
   `_integrate_generic(step_fn, …, order, …)` handles direction‑aware
   integration, step‑size PI control, and returns time‑state history arrays.
   The single implementation is reused by every method.

5. **Public API**  
   - `integrate_rk45(f, t0, y0, t_span, …)`  
   - `integrate_dop853(f, t0, y0, t_span, …)`  
   - `integrate_dp8(f, t0, y0, t_span, …)`  
   Each entry point calls `_integrate_generic` with the appropriate stepper
   and order.

6. **Batch parallelism**  
   `integrate_dp8_batch` uses `prange` to propagate multiple independent
   initial conditions, calling `integrate_dp8` internally.

7. **Module location**  
   `mission_sim/utils/propagators/rk.py` (renamed from `base.py` to leave room
   for future propagator families such as symplectic or multistep methods).

8. **Backward compatibility**  
   The legacy name `integrate_dp8` is preserved; its implementation is
   upgraded to an 8‑th‑order method (currently DOP853).  Existing code that
   imports and calls `integrate_dp8` continues to work without modification.

## Consequences

- **Positive:**  
  - Adding a new RK method requires only a new `RKTable` and a call to the
    factory; control logic is completely reused.  
  - All steppers are fully compiled with fixed loop bounds, maximising
    performance.  
  - The module is small (< 500 lines), easy to review, and clearly separates
    data from algorithm.

- **Negative:**  
  - The DP8(7) placeholder is not yet the genuine 13‑stage 8‑order pair.
    Temporary aliasing to DOP853 slightly reduces the achievable accuracy,
    though it still offers 8‑th order.  A future ADR will replace it with
    verified coefficients.

- **Risks:**  
  - If existing code depends on the numerical behaviour of the old DP5(4)
    `integrate_dp8`, the solution may change slightly.  Tests (e.g., the
    analytical ephemeris tests) already pass with the new implementation.

- **Transition plan:**  
  1. Rename file and update imports throughout the codebase (completed).  
  2. Monitor test suites for any subtle integration error.  
  3. Once genuine DP8(7) coefficients are obtained, replace `TABLE_DP8` and
     increment the ADR.

## References

- Dormand‑Prince coefficients – Hairer, Nørsett, Wanner, *Solving Ordinary
  Differential Equations I*, Springer.  
- DOP853 Fortran source – `dop853.f` (E. Hairer, G. Wanner).  
- RK5(4) Butcher tableau – classical values from Dormand & Prince (1980).
