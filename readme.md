aloha

# What I Learned

[English](readme.md) · [简体中文](README.zh-CN.md)

- The basic Git workflow is: edit files locally → `git add` to stage them → `git commit` to save a snapshot → `git push` to publish commits.
- A branch is a label pointing to a commit. You can work on a new branch and decide later whether to merge it. `git merge` merges the specified branch into the branch currently checked out.
- Environments can be managed with Conda or uv. This project uses uv for a faster, simpler setup. Running code with uv creates a local `.venv` directory; it should not be pushed to Git, so exclude it in `.gitignore`.
- The project records dependencies in `pyproject.toml` and `uv.lock`, and runs code with `uv run`.
- Hugging Face Hub provides `load_dataset` for datasets and `from_pretrained` for pretrained models. Downloads are cached under `~/.cache/huggingface` and do not need to be committed. When network access is a problem, a mirror can be selected with `export HF_ENDPOINT=https://hf-mirror.com`.
- Transfer learning: replace ResNet-18's 1,000-class classification head with a new 10-class head while retaining pretrained features. Because the new head has not been trained, accuracy is roughly random, around 10%, as expected for this assignment.
- I also explored Hugging Face's `Trainer` for training and fine-tuning, `pipeline()` for one-line inference, and uploading models to the Hub.
- To inspect older versions without modifying them, use `git checkout` carefully. Avoid potentially destructive operations such as `git reset` when you only intend to look at history.
