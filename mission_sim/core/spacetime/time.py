"""
Hey! Time conversion made easy.

Provides bidirectional conversion between UTC strings (ISO 8601) and the
following time scales:
  - TAI (International Atomic Time)
  - TT  (Terrestrial Time)
  - TDB (Barycentric Dynamical Time)
  - Julian Date (UTC)
  - Unix timestamp
  - Smooth UTC seconds (leap‑second‑free, referenced to J2000.0 UTC)

All internal time representations are continuous seconds relative to
the J2000.0 epoch, consistent with the MCPC unified time axis.

Leap‑second data are loaded from the local static file ``Leap_Second.dat``
instead of being hard‑coded.  If the file is missing or out‑of‑date,
a diagnostic message will be emitted at import time instructing the user
to run ``tools/update_leap_second.py``.

.. seealso::
   ADR-0004 (docs/design/adr/0004-leap-second-data-management.md)
   for the design rationale behind the external leap‑second file and the
   freshness check.
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Epoch constants
# ---------------------------------------------------------------------------
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # J2000.0 UTC
J2000_JD = 2451545.0   # Julian Date of J2000.0 (TT epoch, commonly used as reference)
TAI_OFFSET_AT_J2000 = 32.0  # TAI - UTC at J2000.0 UTC

# ---------------------------------------------------------------------------
# Path to the leap‑second data file
# ---------------------------------------------------------------------------
_LEAP_FILE_PATH = os.path.join(os.path.dirname(__file__), "Leap_Second.dat")

# ---------------------------------------------------------------------------
# Load leap‑second table from the local file (executed once at import)
# ---------------------------------------------------------------------------
def _parse_leap_second_file(path: str) -> list:
    """
    Parse the Leap_Second.dat file and return a list of leap‑second
    effective dates (as UTC datetime objects), sorted in ascending order.

    File format example::

        41317.0    1  1 1972       10

    Columns 2‑4 are day, month, year; column 5 is the TAI‑UTC value (which
    is ignored by this function).
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                day = int(parts[1])
                month = int(parts[2])
                year = int(parts[3])
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                events.append(dt)
            except (ValueError, IndexError):
                continue
    events.sort()
    return events


def _check_file_freshness(path: str) -> None:
    """
    Verify whether the local leap‑second file is current with respect
    to the half‑yearly Bulletin C cycle.

    If the file's modification time is earlier than the start of the
    current half‑year period (1 January or 1 July), a warning is printed
    to stderr recommending that the file be updated.

    .. seealso::
       ADR-0004 for the design of this freshness check.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return   # file doesn't exist, handled elsewhere

    file_mtime = datetime.fromtimestamp(mtime, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    # Bulletin C half‑year periods: 1 January … 30 June, 1 July … 31 December
    if now.month >= 7:
        period_start = datetime(now.year, 7, 1, tzinfo=timezone.utc)
    else:
        period_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)

    if file_mtime < period_start:
        print(
            f"\N{warning sign}  Leap‑second file '{path}' may be outdated "
            f"(last modified {file_mtime.date()}).\n"
            "   Please run 'python tools/update_leap_second.py' to obtain "
            "the latest data.",
            file=sys.stderr,
        )


# Actual loading and freshness check (performed once at import time)
if not os.path.exists(_LEAP_FILE_PATH):
    print(
        f"\N{warning sign}  Leap‑second file '{_LEAP_FILE_PATH}' not found.\n"
        "   Please run 'python tools/update_leap_second.py' to initialise "
        "the local data file.",
        file=sys.stderr,
    )
    _LEAP_SECONDS_DATES = []
else:
    _LEAP_SECONDS_DATES = _parse_leap_second_file(_LEAP_FILE_PATH)
    _check_file_freshness(_LEAP_FILE_PATH)


# ---------------------------------------------------------------------------
# Leap‑second management
# ---------------------------------------------------------------------------
def leap_seconds(utc_time: datetime) -> int:
    """
    Return the cumulative number of leap seconds (TAI − UTC) that have
    occurred on or before the given UTC datetime.

    For times earlier than the first leap second this function returns 0.
    """
    cnt = 0
    for d in _LEAP_SECONDS_DATES:
        if d <= utc_time:
            cnt += 1
        else:
            break
    return cnt


def add_leap_second(date_str: str) -> None:
    """
    Temporarily insert a leap‑second date into the in‑memory table
    (does **not** modify the data file, only valid for the current process).

    Parameters
    ----------
    date_str : str
        ISO‑format date string, e.g. ``"2026-01-01"``.
    """
    new_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    _LEAP_SECONDS_DATES.append(new_date)
    _LEAP_SECONDS_DATES.sort()


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------
def utc_string_to_utc_smooth(utc_iso: str) -> float:
    """Convert a UTC ISO string to smooth UTC seconds (leap‑second‑free)
    relative to J2000.0 UTC."""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return (dt - J2000_UTC).total_seconds()


def utc_smooth_to_utc_string(utc_smooth_sec: float) -> str:
    """Inverse of `utc_string_to_utc_smooth`: convert smooth UTC seconds
    back to an ISO UTC string."""
    dt = J2000_UTC + timedelta(seconds=utc_smooth_sec)
    return dt.isoformat()


def utc_string_to_tai(utc_iso: str) -> float:
    """Convert UTC ISO string to TAI seconds since J2000.0 TAI."""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    utc_smooth = (dt - J2000_UTC).total_seconds()
    leap = leap_seconds(dt)
    return utc_smooth + leap - TAI_OFFSET_AT_J2000


def tai_to_utc_string(tai_sec: float) -> str:
    """Inverse of `utc_string_to_tai`."""
    for leap in range(0, 60):
        utc_smooth = tai_sec - leap + TAI_OFFSET_AT_J2000
        dt = J2000_UTC + timedelta(seconds=utc_smooth)
        if leap_seconds(dt) == leap:
            return dt.isoformat()
    raise ValueError(f"Unable to find a valid UTC time for TAI = {tai_sec}")


def utc_string_to_tt(utc_iso: str) -> float:
    """Convert UTC ISO string to TT seconds since J2000.0 TT."""
    tai = utc_string_to_tai(utc_iso)
    return tai + 32.184   # fixed offset between TAI and TT


def tt_to_utc_string(tt_sec: float) -> str:
    """Inverse of `utc_string_to_tt`."""
    tai = tt_sec - 32.184
    return tai_to_utc_string(tai)


def utc_string_to_tdb(utc_iso: str) -> float:
    """
    Convert UTC ISO string to TDB seconds since J2000.0 TDB using a
    simplified analytic approximation (Fairhead & Bretagnon, 1990);
    accuracy < 1 µs.
    """
    tt = utc_string_to_tt(utc_iso)
    t_tt = tt / 86400.0 / 36525.0       # Julian centuries since J2000 TT
    g = (357.528 + 35999.05 * t_tt) * math.radians(1)
    tdb_offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
    return tt + tdb_offset


def tdb_to_utc_string(tdb_sec: float) -> str:
    """Inverse of `utc_string_to_tdb` using a fixed‑point iteration."""
    tt = tdb_sec
    for _ in range(5):
        t_tt = tt / 86400.0 / 36525.0
        g = (357.528 + 35999.05 * t_tt) * math.radians(1)
        offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
        tt = tdb_sec - offset
    return tt_to_utc_string(tt)


def utc_string_to_jd(utc_iso: str) -> float:
    """Convert UTC ISO string to continuous UTC Julian Date."""
    utc_smooth = utc_string_to_utc_smooth(utc_iso)
    return J2000_JD + utc_smooth / 86400.0


def jd_to_utc_string(jd: float) -> str:
    """Inverse of `utc_string_to_jd`."""
    utc_smooth = (jd - J2000_JD) * 86400.0
    return utc_smooth_to_utc_string(utc_smooth)


def utc_string_to_unix(utc_iso: str) -> float:
    """Convert UTC ISO string to Unix timestamp (POSIX seconds)."""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def unix_to_utc_string(unixtime: float) -> str:
    """Inverse of `utc_string_to_unix`."""
    dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# New high-performance numeric conversions (no string intermediate)
# ---------------------------------------------------------------------------
def utc_smooth_to_tdb(utc_smooth_sec: float) -> float:
    """Convert smooth UTC seconds to TDB seconds since J2000.0.

    This function avoids an ISO‑string round‑trip and is suitable for
    high‑rate calls (e.g. inside N‑body propagation).
    """
    dt = J2000_UTC + timedelta(seconds=utc_smooth_sec)
    leap = leap_seconds(dt)
    tai = utc_smooth_sec + leap - TAI_OFFSET_AT_J2000
    tt = tai + 32.184                                   # TT = TAI + 32.184 s
    t_tt = tt / 86400.0 / 36525.0                      # Julian centuries since J2000 TT
    g = (357.528 + 35999.05 * t_tt) * math.radians(1)
    tdb = tt + 0.001658 * math.sin(g + 0.0167 * math.sin(g))
    return tdb


def tdb_to_utc_smooth(tdb_sec: float) -> float:
    """Inverse of `utc_smooth_to_tdb` – fixed‑point iteration followed by
    a leap‑second loop."""
    # fixed‑point solve for TT
    tt = tdb_sec
    for _ in range(5):
        t_tt = tt / 86400.0 / 36525.0
        g = (357.528 + 35999.05 * t_tt) * math.radians(1)
        offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
        tt = tdb_sec - offset
    tai = tt - 32.184
    # brute‑force search for the correct leap second count
    for leap in range(0, 60):
        utc_smooth = tai + TAI_OFFSET_AT_J2000 - leap
        dt = J2000_UTC + timedelta(seconds=utc_smooth)
        if leap_seconds(dt) == leap:
            return utc_smooth
    raise ValueError(f"Cannot convert TDB {tdb_sec} to UTC smooth")


# ---------------------------------------------------------------------------
# Legacy compatibility functions (mirror aliases from astro.py)
# ---------------------------------------------------------------------------
def utc2tai(utc_jd: float) -> float:
    """Legacy: convert UTC Julian date to TAI Julian date."""
    utc_smooth = (utc_jd - J2000_JD) * 86400.0
    dt = J2000_UTC + timedelta(seconds=utc_smooth)
    leap = leap_seconds(dt)
    return utc_jd + leap / 86400.0


def utc2tdt(utc_jd: float) -> float:
    """Legacy: convert UTC Julian date to TDT (TT) Julian date."""
    return utc2tai(utc_jd) + 32.184 / 86400.0


def utc2tdb(utc_jd: float) -> float:
    """Legacy: convert UTC Julian date to TDB Julian date (simplified)."""
    tdt = utc2tdt(utc_jd)
    jc = (utc_jd - J2000_JD) / 36525.0
    g = 2.0 * math.pi * (357.528 + 35999.05 * jc) / 360.0
    return tdt + 0.001658 * math.sin(g + 0.0167 * math.sin(g)) / 86400.0


def unix2utc(t: float) -> float:
    """Legacy: convert Unix timestamp to UTC Julian day."""
    return t / 86400.0 + 2440587.5


def utc2unix(utc_jd: float) -> float:
    """Legacy: convert UTC Julian day to Unix timestamp."""
    return (utc_jd - 2440587.5) * 86400.0
