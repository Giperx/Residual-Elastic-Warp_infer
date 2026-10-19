"""Runtime compatibility shims for the Residual-Elastic-Warp models.

Older releases of this repository (and the upstream IHN code base) called
``torch.gesv`` / ``torch.solve`` which were removed in recent PyTorch. The
models package relies on that operator, so the shim must be installed *before*
``models`` is imported.

Only ``torch.gesv`` is patched here. ``torch.inverse`` is intentionally left
untouched; the CUDA-solver workaround belongs in the caller, not in a global
monkey patch.
"""
import torch


def gesv_patch(B, A):
    """Robust replacement for the removed ``torch.gesv(B, A)``.

    Behaviour mirrors the remote eval driver:

    1. try ``torch.linalg.solve(A, B)`` (note the argument order swap),
    2. on a singular matrix retry with epsilon Tikhonov regularisation,
    3. fall back to a least-squares solve,
    4. return ``(solution, A)`` to mimic the historic ``(X, LU)`` tuple.
    """

    def solve_ax_b(matrix, rhs):
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
            return torch.linalg.solve(matrix, rhs)
        solution, _ = torch.solve(rhs, matrix)
        return solution

    try:
        solution = solve_ax_b(A, B)
    except RuntimeError:
        try:
            epsilon = 1e-6
            eye = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).expand_as(A)
            solution = solve_ax_b(A + epsilon * eye, B)
        except RuntimeError:
            if hasattr(torch, "linalg") and hasattr(torch.linalg, "lstsq"):
                solution = torch.linalg.lstsq(A, B).solution
            else:
                solution, _ = torch.lstsq(B, A)

    # Historic torch.gesv returned (X, LU). A stands in for the unused LU.
    return solution, A


def install():
    """Install the gesv shim on ``torch`` only when it is missing."""
    if not hasattr(torch, "gesv"):
        torch.gesv = gesv_patch
    return torch.gesv
