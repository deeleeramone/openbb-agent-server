"""Range and shape validation helpers shared by model providers."""

from __future__ import annotations


def check_range(name: str, value: float | None, lo: float, hi: float) -> None:
    """Validate that a value falls within an inclusive numeric range.

    A ``None`` value is treated as unset and passes validation.

    Parameters
    ----------
    name : str
        Human-readable parameter name, used in the error message.
    value : float or None
        The value to check. ``None`` skips the check.
    lo : float
        Inclusive lower bound.
    hi : float
        Inclusive upper bound.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` and falls outside ``[lo, hi]``.
    """
    if value is None:
        return
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi} (got {value})")


def check_min(name: str, value: int | None, lo: int) -> None:
    """Validate that an integer meets an inclusive lower bound.

    A ``None`` value is treated as unset and passes validation.

    Parameters
    ----------
    name : str
        Human-readable parameter name, used in the error message.
    value : int or None
        The value to check. ``None`` skips the check.
    lo : int
        Inclusive minimum allowed value.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` and is less than ``lo``.
    """
    if value is None:
        return
    if value < lo:
        raise ValueError(f"{name} must be >= {lo} (got {value})")
