"""SM120 kernel recipe."""

from .mainloop import SolAttnForwardSm120


def make_kernel(
    *,
    debug_route_trace: bool = False,
    prefetch_first_exact_k: bool = True,
    prefetch_next_route_k: bool = True,
    external_route: bool = False,
    hybrid_route: bool = False,
    exact_only: bool = False,
    export_route: bool = False,
    force_local_blocks: bool = True,
):
    return SolAttnForwardSm120(
        debug_route_trace=debug_route_trace,
        prefetch_first_exact_k=prefetch_first_exact_k,
        prefetch_next_route_k=prefetch_next_route_k,
        external_route=external_route,
        hybrid_route=hybrid_route,
        exact_only=exact_only,
        export_route=export_route,
        force_local_blocks=force_local_blocks,
    )


__all__ = ["make_kernel"]
