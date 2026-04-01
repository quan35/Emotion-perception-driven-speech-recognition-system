# Experiment Protocol

This reference captures the current experiment protocol that the skill should follow when reasoning about training, comparison, and promotion tasks in this repository.

## Mainline scope

The mainline system is the shared Whisper-based model implemented around `models/whisper_emotion.py`, executed through `scripts/train_shared.py` and typically launched by `scripts/train_shared_tmux.sh`, visualized in `notebooks/04_train_shared.ipynb`, and used at inference time through `inference/pipeline.py`. The early `CNN+BiLSTM+Attention` route is retained for historical contrast and should not be treated as the default experimental baseline unless the user explicitly requests it.

## Monitoring rule

Current script behavior in `scripts/train_shared.py` selects the best epoch using the monitor chain `val_subset_mean_uar -> val_uar -> val_loss`. This is stricter than a single-metric monitor and must be preserved when auditing results. If `configs/config.yaml` or older prose mentions only `subset_mean_uar` or `val_uar`, report the discrepancy instead of flattening it.

## Expected outputs for a completed shared-model run

For an experiment run named `<run_name>`, expect the following files under `checkpoints/`:

- `<run_name>.pth`
- `<run_name>_summary.json`
- `<run_name>_history.npz`
- `<run_name>_curves.png`
- `<run_name>_confusion_matrix.png`

For a protocol family or grouped comparison, there may also be:

- `<experiment_stem>_aggregate.json`

## Metrics to cite by default

When summarizing or comparing runs, prefer this order:

1. `selected_val_subset_mean_uar`
2. `selected_val_uar`
3. `test_subset_mean_uar`
4. `uar`
5. `macro_f1`
6. `test_acc`

Use `best_val_*` values only when discussing the shape of the optimization trajectory. Use `selected_val_*` when discussing the epoch actually chosen as the best checkpoint.

## Default execution pattern

When the user asks to run the shared-model mainline, prefer the script-first helper:

```bash
bash scripts/train_shared_tmux.sh --session-name train_shared_<tag> -- --profile cuda_4090_mainline --norm <derf|dyt>
```

If the current machine has no GPU or the user only wants a protocol preflight, prefer:

```bash
bash scripts/train_shared_tmux.sh --session-name train_shared_audit_<tag> -- --profile cpu_preflight --audit-only
```

Use `notebooks/04_train_shared.ipynb` after the run for artifact analysis and visualization rather than for executing the training protocol itself.

## Promotion rule

Only recommend updating `configs/config.yaml` `paths.best_shared_model` after all of the following are true:

- The checkpoint file exists.
- The summary and history files exist.
- The training curves and confusion matrix exist.
- The chosen best epoch is consistent with the stored history.
- The reported metrics in the paper or README will not drift from the new default path.

## Comparison language

When the user asks for a narrative summary, prefer language like:

"在保持其余训练协议不变的前提下，`Derf` 配置在 `selected_val_subset_mean_uar` 上高于 `DyT`，同时测试集 `uar` 亦保持优势，因此该改动更适合作为当前主线默认配置。与此同时，若仅有辅助指标提升而主监控链路下降，则应将该结果表述为权衡而非整体改进。"

If the task updates a report file, rewrite the result as complete academic paragraphs rather than bullet lists.
