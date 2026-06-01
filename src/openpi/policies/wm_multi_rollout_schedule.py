"""Pure helpers for multi-round WM+AE chunk indexing (no JAX)."""


def wm_multi_rollout_max_rounds_schedule(*, h: int, overlap: int, delta_idx: int) -> int:
    """Largest N (WM+AE cycles) with ``overlap + (N-1)*delta_idx <= h`` (fixed multi-round feasibility)."""
    if delta_idx < 1:
        return 1
    return max(1, (h - overlap) // delta_idx + 1)


def wm_multi_rollout_max_rounds_user_cap(*, h: int, overlap: int, delta_idx: int) -> int:
    """Largest N with ``2*overlap + (N-1)*delta_idx < h`` (strict stricter cap for adaptive κ tests)."""
    if delta_idx < 1 or h <= 2 * overlap:
        return 1
    return max(1, (h - 2 * overlap - 1) // delta_idx + 1)


def wm_multi_rollout_adaptive_max_rounds(*, h: int, overlap: int, delta_idx: int) -> int:
    """Minimum of schedule-feasible rounds and the ``2*overlap + (N-1)*Δ < H`` cap."""
    return max(
        1,
        min(
            wm_multi_rollout_max_rounds_schedule(h=h, overlap=overlap, delta_idx=delta_idx),
            wm_multi_rollout_max_rounds_user_cap(h=h, overlap=overlap, delta_idx=delta_idx),
        ),
    )
