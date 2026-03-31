---
name: experiment-runner
description: Use when the user asks to run, resume, improve, optimize, compare, audit, critique, troubleshoot, or summarize experiments in this repository, especially requests such as `实验改进`, `实验优化`, `消融`, `对比实验`, `比较 Derf 和 DyT`, `调整 freeze_strategy / training_mode / norm`, `看哪个 checkpoint 更好`, `是否该切换默认模型`, `分析 summary.json / history.npz`, `解释混淆矩阵`, `复盘一次训练`, `根据本地结果写实验分析`, Whisper + Norm-Free Transformer shared-model runs, notebook executions in notebooks/04_train_shared.ipynb, config changes in configs/config.yaml, or checkpoint promotion.
---

# Experiment Runner

Use this skill for repository-specific experiment execution, comparison, artifact auditing, and local-evidence analysis.

## Use when

- The user asks to run or resume a training experiment.
- The user asks how to improve, optimize, stabilize, or debug a local experiment result.
- The user asks to compare `Derf / DyT`, `freeze_strategy`, `training_mode`, seeds, or subset sampling behavior.
- The user asks whether a checkpoint is complete, reproducible, or ready to become the default model.
- The user asks to reconcile paper or thesis claims with `summary.json`, `history.npz`, or checkpoint files.
- The user asks for experiment-backed interpretation rather than literature-backed interpretation.

## Common trigger phrasing

- `帮我做实验改进`
- `这个结果还能怎么优化`
- `帮我设计消融实验`
- `比较一下 Derf 和 DyT`
- `看看这个 checkpoint 值不值得切默认`
- `帮我分析 summary.json / history.npz`
- `根据本地实验结果写一段分析`
- `这次实验为什么 uar 上去了但是主指标没提升`

## Repository assumptions

- The mainline experiment entry is `notebooks/04_train_shared.ipynb`.
- `notebooks/03_train_emotion.ipynb` is historical and should only be used when the user explicitly wants the early exploration route.
- Emotion labels are fixed to `happy`, `angry`, `sad`, `neutral`, `fear`, `surprise`. Do not silently remap labels or alter class count.
- `data/` may be a symlink or mounted path. Do not assume raw data is tracked in git.
- Current executable monitoring logic in `notebooks/04_train_shared.ipynb` uses `val_subset_mean_uar -> val_uar -> val_loss` for best-epoch selection. `configs/config.yaml` still exposes `training.best_metric: subset_mean_uar` as a shorthand. When these disagree, treat notebook behavior plus generated artifacts as execution truth and call out the mismatch explicitly.

## Read order

1. Read `AGENTS.md`.
2. Read `configs/config.yaml`.
3. If the task is mainline training or comparison, inspect `notebooks/04_train_shared.ipynb`.
4. If the task executes notebooks, inspect `scripts/run_notebook_tmux.sh`.
5. If the task is about an existing run, inspect the corresponding `checkpoints/*_summary.json` and `checkpoints/*_history.npz`.
6. If the task updates a report or paper, keep the repository writing constraints in force. Report files must use complete academic paragraphs rather than bullet lists.

## Interoperability

Use this skill as the local-evidence source of truth, then combine it with other skills when the task expands beyond raw experiment handling.

- Add `paper-writer` when the user wants the checked experiment results rewritten as Chinese academic prose, methods text, result analysis, discussion, or figure captions.
- Add `peer-review` when the user wants to judge whether the experiment design, metric choice, fairness, reproducibility, or conclusions are defensible.
- Add `literature-review` when the user wants to position a local result against external work or state-of-the-art trends.
- Add `citation-management` when the user needs DOI, BibTeX, metadata verification, or reference cleanup tied to the experiment discussion.
- If the request is broad and the correct combination is unclear, keep `ser-research-router` active and let it select the minimal set of companion skills.

## Default workflow

### Running a mainline experiment

1. Identify the single primary variable being changed. If the request mixes multiple variables, split the experiment design so attribution remains clear.
2. Snapshot the config values that matter most: `shared_model.training_mode`, `shared_model.norm`, `shared_model.freeze_strategy`, `training.seed`, `training.seed_sweep`, `training.subset_sampling_mode`, `training.subset_epoch_targets`, `training.subset_epoch_caps`, and `paths.best_shared_model`.
3. Prefer the existing notebook-first workflow unless the user explicitly asks to script the training entry. If executing the notebook, prefer:

```bash
bash scripts/run_notebook_tmux.sh notebooks/04_train_shared.ipynb <session_name>
```

4. Record the tmux session name, executed notebook path under `runs/`, and log path under `logs/`.
5. After the run, audit the produced artifacts before drawing conclusions.

### Auditing a completed run

Use the bundled script:

```bash
python .agents/skills/experiment-runner/scripts/audit_shared_experiment.py \
  --run-name <experiment_run_name>
```

If the user wants to promote the run to the default inference checkpoint, add:

```bash
python .agents/skills/experiment-runner/scripts/audit_shared_experiment.py \
  --run-name <experiment_run_name> \
  --check-config-default
```

### Comparing experiments

- Compare the same protocol family only. Do not compare the early exploration model against the mainline model as if they were the same pipeline.
- Prefer `selected_val_subset_mean_uar`, `selected_val_uar`, `test_subset_mean_uar`, `uar`, `macro_f1`, and `test_acc` from `summary.json`, and verify them against `history.npz` when the result matters.
- If only one metric improves while the main monitor chain degrades, describe the tradeoff rather than declaring a blanket improvement.
- When the user asks for a paper or report update, keep this skill responsible for the evidence selection and let `paper-writer` handle the final academic phrasing.

### Promoting a checkpoint

- Do not update `configs/config.yaml` `paths.best_shared_model` until the run has been audited.
- Mention the exact checkpoint filename and the rationale for promotion.
- If the best checkpoint changed because of a protocol tweak, state whether historical comparisons remain valid.

## Required artifact checks

- `*_summary.json` exists and contains the expected monitor and metric fields.
- `*_history.npz` exists and includes `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, `val_uar`, and `val_subset_mean_uar`.
- The checkpoint file referenced by the summary exists.
- The training curves and confusion matrix referenced by the summary exist.
- `best_epoch`, `selected_val_*`, and `best_val_*` fields are numerically consistent with `history.npz`.

## Ground-truth policy

- Prefer executable truth over stale prose. In this repository that means current notebook logic, actual generated artifacts, and current config values outrank older narrative descriptions.
- If documentation and artifacts disagree, do not silently normalize them. Surface the mismatch and explain which source was treated as ground truth.

## References

- Read `references/protocol.md` for the repo-specific protocol, expected outputs, and comparison language.
