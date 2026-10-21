"""Temporal index selection shared by Ref2VA comparison scripts."""

CHUNK_LATENT_FRAMES = 5


def _chunk_phase_indices(
    dense_latent_frames: int,
    normal_tokens: int,
    single_normal_index: int = 0,
    normal_indices: str = "",
    head_tokens: int = 1,
) -> list[int]:
    """Keep each chunk head plus the requested normal-token phase pattern.

    Two-normal mode retains normal indices 0 and 2. One-normal mode defaults
    to index 0, while allowing experiments to select a different phase.
    """
    if dense_latent_frames <= 0:
        raise ValueError("dense latent frame count must be positive")
    if normal_tokens not in (0, 1, 2, 3, 4):
        raise ValueError("chunk_normal_tokens must be between 0 and 4")
    if single_normal_index not in (0, 1, 2, 3):
        raise ValueError("chunk_single_normal_index must be 0, 1, 2, or 3")
    if head_tokens not in (0, 1):
        raise ValueError("chunk_head_tokens must be 0 or 1")
    if normal_indices.strip():
        try:
            phases = [int(value.strip()) for value in normal_indices.split(",")]
        except ValueError as exc:
            raise ValueError("chunk_normal_indices must be comma-separated integers") from exc
        if len(phases) != normal_tokens:
            raise ValueError("chunk_normal_indices count must equal chunk_normal_tokens")
        if len(set(phases)) != len(phases) or any(phase not in range(4) for phase in phases):
            raise ValueError("chunk_normal_indices must be unique values from 0 through 3")
        phases.sort()
    elif normal_tokens == 1:
        phases = [single_normal_index]
    elif normal_tokens == 2:
        phases = [0, 2]
    else:
        phases = list(range(normal_tokens))
    selected: list[int] = []
    for chunk_start in range(0, dense_latent_frames, CHUNK_LATENT_FRAMES):
        if head_tokens:
            selected.append(chunk_start)
        normal = list(
            range(
                chunk_start + 1,
                min(chunk_start + CHUNK_LATENT_FRAMES, dense_latent_frames),
            )
        )
        selected.extend(normal[phase] for phase in phases if phase < len(normal))
    return selected
