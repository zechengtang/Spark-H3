"""Blackwell kernel entry."""

from .mainloop import SolAttnForwardSm100, forward


def make_kernel(
    *,
    external_route: bool = False,
    hybrid_route: bool = False,
    exact_only: bool = False,
    export_route: bool = False,
    force_local_blocks: bool = True,
):
    return SolAttnForwardSm100(
        external_route=external_route,
        hybrid_route=hybrid_route,
        exact_only=exact_only,
        export_route=export_route,
        force_local_blocks=force_local_blocks,
    )


__all__ = ["forward", "make_kernel"]
