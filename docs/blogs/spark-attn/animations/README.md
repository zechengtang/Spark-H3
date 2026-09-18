# Spark-Attn animations

Two self-contained HTML/SVG animations, embedded in the research blog:

- `spark-reblock.html`: a balanced 16-token toy example. The same positive
  attention entries are permuted into coherent blocks. Selecting one of four
  key blocks per query block captures 25.0% of attention mass before reordering
  and 94.8% afterward. The animation first splits the 16-token parent into two 8-token children,
  then splits each child into two 4-token leaves. The tree tracks token IDs.
  These binary partitions use predetermined toy preferences to illustrate
  recursion; production uses learned similarity, configured fanout and
  64-token leaves. The mass numbers are toy arithmetic, not model quality.
- `spark-reweight.html`: four compressed tokens with logits [-2,-1,1,2] and
  scalar values [-1,-0.5,0.5,1], plus a fixed exact branch. Restoring both mass
  and the weighted value recovers the reference output at the shared query.
  The two corrections are animated separately for explanation. Nearby queries
  still use the algorithm's approximation.

Each page works offline when opened directly and includes Play/Pause, Replay
and timeline scrubbing. Playback stops after one pass, respects reduced-motion
preferences on load, and pauses while the visualization is off-screen.

Rebuild both generated pages with:

```bash
python docs/blogs/landmark_tree_v2/animations/build.py
```

The blog preview explicitly serves these two files under `/animations/`.
No external script, stylesheet, font, model weights or video data is required.
Browser verification covers start/intermediate/end states, the toy arithmetic,
play/pause/replay, desktop and mobile layouts, and the live blog embeds.
Screenshots are stored outside Git at `/tmp/h3_spark_animations/`.
