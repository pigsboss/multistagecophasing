# ADR-0004: Leap‑Second Data Management and Time Scale Conversions

## Status

Accepted

## Context

Accurate time scale conversions (UTC, TAI, TT, TDB) are essential for
orbit propagation and spacecraft simulations. The International Earth
Rotation and Reference Systems Service (IERS) periodically announces
leap seconds, which must be incorporated into the time system.

The original `astro.py` hard‑coded the leap‑second table inside the
source file, making updates tedious and error‑prone. To improve
maintainability and ensure that the data can be refreshed independently
of the source code, we decided to externalise the leap‑second list.

## Decision

### 1. External leap‑second data file

Leap‑second information is stored in a plain text file named
`Leap_Second.dat` located at `mission_sim/core/spacetime/`. This file
follows the format provided by the Paris Observatory IERS Earth
Orientation Centre:

