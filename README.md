# NYU Building LLM Reasoners Assignment 3: Alignment

This assignment is adapted from Stanford CS336 Assignment 5 ([original repository](https://github.com/stanford-cs336/)). All credit for its
development goes to the Stanford course staff. This README and all of the following code are adapted from theirs.

For a full description of the assignment, see the assignment handout at
[a3.pdf](https://gregdurrett.github.io/courses/sp2026/a3.pdf)

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### pyproject-mac.toml

We've included -mac variants of `pyproject.toml` and `uv.lock`. These do not include vllm, which will limit your options for fast inference for
real experiments, but allow you to complete much of the preliminary testing of models locally.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).
