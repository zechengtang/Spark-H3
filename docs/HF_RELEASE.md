# Hugging Face release checkout

The development repository is the source of truth for shared code. The separate
Hugging Face repository at `https://huggingface.co/Aazeus/Spark-H3` contains its
own Git history, LFS configuration, and demo videos. It is a publication target,
not another place to edit shared code.

On a new server, clone the development repository and, when a release is needed,
clone the existing Hugging Face repository too:

```bash
git clone https://github.com/zechengtang/Spark-H3.git Spark-H3
git clone https://huggingface.co/Aazeus/Spark-H3 Spark-H3-huggingface-release
python Spark-H3/tools/sync_hf_release.py Spark-H3-huggingface-release
```

The command previews changes using Git-tracked files in the current working
tree. Applying changes requires both checkouts to have no uncommitted tracked
changes. It copies development files except
`.gitignore`, `.github/`, and the GitHub README. `README_HF.md` maps to the HF
`README.md`. The HF checkout keeps its own `.gitattributes` and
`comfyui/demos/` files. The tool does not delete HF-only files, commit, push, or
copy ignored model weights. Existing HF-only files outside those paths are
listed for review.

Before publishing, review the preview and reconcile the model card. The current
`README_HF.md` includes VDN sample links that are absent from the local HF
checkout; the published card has Larryvrh comparisons absent from that source
file. Do not replace the card until the intended samples and linked media have
been checked. To copy shared files after review, run:

```bash
python Spark-H3/tools/sync_hf_release.py Spark-H3-huggingface-release --apply
```

Once the model card is reconciled, add `--replace-card` to copy it as the HF
`README.md`. Inspect `git -C Spark-H3-huggingface-release diff` and `status`
before committing or pushing the HF repository. Repeat the preview after the
copy; shared files should then show no differences.
