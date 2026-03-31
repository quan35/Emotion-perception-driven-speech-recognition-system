---
name: paper-writer
description: Use when the user asks to write, revise, expand, polish, restructure, or localize thesis or paper content for this repository in Chinese academic prose, especially requests such as `写相关工作`, `补方法部分`, `写实验设置`, `写结果分析`, `写讨论`, `写结论`, `写图表说明`, `润色摘要`, `改成论文语言`, `扩写成中文学术段落`, `根据本地实验写一段结果分析`, `把图表结论写出来`, or when manuscript text must stay consistent with local checkpoints, summary.json, history.npz, and the repository's report-writing constraints.
---

# Paper Writer

Use this skill for manuscript and thesis writing tasks in this repository.

## Use when

- The user asks to draft or revise `相关工作`, `方法`, `实验设置`, `结果分析`, `讨论`, `结论`, `图表说明`, or `摘要`.
- The user asks for Chinese academic paragraphs rather than outline bullets.
- The user asks to turn local experiment outputs into paper-ready narrative.
- The user asks to reconcile manuscript wording with repository evidence.

## Common trigger phrasing

- `把这段改成论文语言`
- `帮我写一段中文学术表述`
- `补一段相关工作`
- `根据实验结果写分析`
- `把这个表格写成论文里的描述`
- `帮我润色摘要和结论`
- `把方法部分扩写完整`
- `给我一段能直接放论文里的结果分析`

## Scope

- Default language is the user's language. For this repository, most paper-facing output should remain Chinese unless the user explicitly asks for English.
- Treat report files as academic prose only. Do not use unordered or numbered lists inside report documents.
- Keep claims conservative. Do not imply causal or general superiority unless local evidence actually supports it.

## Read order

1. Read `AGENTS.md`.
2. If writing about local experiments, read `configs/config.yaml` and inspect the relevant `checkpoints/*_summary.json` and `checkpoints/*_history.npz`.
3. If the task involves experiment comparison or checkpoint promotion, also use the local `experiment-runner` skill.
4. If the task is related work or state-of-the-art, also use `literature-review`.
5. If the task needs DOI, BibTeX, metadata verification, or reference cleanup, also use `citation-management`.

## Interoperability

This skill owns the final manuscript phrasing, not the upstream evidence collection.

- Add `experiment-runner` before writing any local result analysis, hyperparameter comparison, ablation discussion, checkpoint justification, or figure caption grounded in repository outputs.
- Add `literature-review` before writing related work, background surveys, state-of-the-art positioning, or research gap discussions.
- Add `citation-management` when the text needs verified citations, BibTeX cleanup, DOI checks, or reference disambiguation.
- Add `peer-review` when the user asks whether the writing overstates the evidence, whether the experimental claims are too strong, or whether the section would survive reviewer scrutiny.
- If the request is broad and no single writing source is obvious, keep `ser-research-router` active so the supporting skills are selected before drafting begins.

## Writing rules

- Prefer complete academic paragraphs with explicit transitions such as `此外`, `同时`, `进一步而言`, `与此相比`, `基于上述结果`, and `综上所述`.
- Use objective claims. Attribute numeric statements to checked artifacts rather than memory.
- Separate `实验事实` from `解释`. First state what changed numerically, then explain the likely reason.
- When discussing model improvements, identify the comparison baseline explicitly.
- When discussing limitations, state boundary conditions rather than generic caveats.

## Section guidance

### Related Work

- Organize by technical route rather than paper-by-paper summaries.
- Suggested axes for this repository include:
  - conventional handcrafted-feature SER
  - CNN/RNN hybrid models
  - self-supervised or pretrained speech encoders
  - Whisper-based shared ASR/SER modeling
- End the section by locating this repository's mainline contribution precisely, not broadly.

### Methods

- Explain the task definition first: joint ASR plus SER, with the SER label space fixed to six classes.
- Distinguish the historical baseline from the current mainline architecture.
- Make the data protocol, speaker-group split, subset sampling, monitoring metric, and checkpoint policy explicit when they are relevant to reproducibility.

### Experiment Setup

- State datasets, label mapping policy, split policy, main hyperparameters, and monitored metrics.
- If `config.yaml` and notebook behavior diverge, treat executable notebook behavior plus artifacts as ground truth and mention the discrepancy carefully if it matters for the text.

### Result Analysis

- Anchor claims to `selected_val_subset_mean_uar`, `selected_val_uar`, `test_subset_mean_uar`, `uar`, `macro_f1`, and `test_acc` when available.
- For local experiment comparisons, prefer `summary.json` values that correspond to the chosen best checkpoint rather than only peak history values.
- Describe tradeoffs explicitly when one metric improves but the primary monitor chain does not.

### Figure and Table Descriptions

- Explain what the figure or table shows, what the main trend is, and what conclusion can and cannot be drawn.
- Do not restate every number in prose if the figure already carries them.
- If the figure comes from local experiment artifacts, verify the filenames and metrics before writing.

## Output policy

- For direct user replies, concise prose is fine.
- For report or thesis files, write polished academic paragraphs only.
- If evidence is missing, say what is unavailable instead of filling gaps with plausible-sounding text.

## References

- Read `references/section_patterns.md` for section-specific patterns and claim boundaries.
