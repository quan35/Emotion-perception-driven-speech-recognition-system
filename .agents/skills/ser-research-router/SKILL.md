---
name: ser-research-router
description: Use when the user asks for research-oriented help in this repository without naming a skill explicitly, especially requests such as `帮我想实验改进`, `帮我优化当前实验`, `做消融`, `判断 checkpoint 是否可靠`, `写论文相关工作`, `把实验结果写成论文语言`, `补最近文献`, `核对引用`, `检查方法是否严谨`, `把本地实验和外部文献一起写成讨论`, or when Codex needs to decide whether to use experiment-runner, peer-review, literature-review, citation-management, and paper-writer alone or together.
---

# SER Research Router

Use this skill to choose the right specialized workflow for research tasks in this repository.

## Use when

- The user asks a research or thesis question but does not specify which skill to use.
- The request could plausibly belong to experiments, methodology critique, literature synthesis, citation verification, or manuscript drafting.
- The user mixes multiple research subtasks in one request and needs routing to a minimal skill set.

## Common trigger phrasing

- `帮我想一下这个课题接下来怎么改进`
- `这个实验下一步怎么做`
- `帮我看看论文这部分该怎么写`
- `我有本地结果，也想补文献`
- `帮我核对参考文献顺便润色正文`
- `这套实验设计严谨吗`
- `我不确定该用哪个 skill`
- `你自己判断该调用哪些 skill`

## Do not use when

- The task is a plain code bugfix or infrastructure change with no experiment or paper component.
- The task is purely UI or deployment work.

## Routing rules

### Route to `experiment-runner`

Use `experiment-runner` when the request is mainly about:

- running or resuming training
- changing `norm`, `freeze_strategy`, `training_mode`, or sampling settings
- comparing local checkpoints
- reading `summary.json`, `history.npz`, `curves`, or `confusion_matrix`
- deciding whether a checkpoint should become the default model

### Add `peer-review`

Use `peer-review` alongside `experiment-runner` when the user is asking for:

- experimental rigor review
- whether the comparison design is fair
- whether metrics, controls, split policy, or claims are defensible
- whether the manuscript overstates local results

### Route to `literature-review`

Use `literature-review` when the request is mainly about:

- related work
- state-of-the-art surveys
- research gaps
- thematic synthesis of prior papers
- building a literature-backed chapter or subsection

### Add `citation-management`

Use `citation-management` when the request includes:

- DOI, PMID, arXiv ID, or BibTeX work
- reference cleanup
- metadata verification
- duplicate citation checks
- matching manuscript references to actual papers

### Route to `paper-writer`

Use `paper-writer` when the request is mainly about:

- drafting or revising Chinese academic paragraphs
- writing methods, experiment setup, results, discussion, or captions
- turning checked local results into manuscript prose

If the section depends on local experiments, combine `paper-writer` with `experiment-runner`.
If the section depends on literature synthesis, combine `paper-writer` with `literature-review`.
If the section depends on verified references, combine `paper-writer` with `citation-management`.

## Free-interoperability policy

- Do not treat routing as a one-skill-only choice. Keep every necessary skill active when the user request genuinely spans multiple evidence sources.
- Prefer additive combinations over serial replacement when the task needs both local experiment evidence and external literature or references.
- Let `experiment-runner` own local metrics and artifacts, `literature-review` own external synthesis, `citation-management` own reference verification, `peer-review` own rigor critique, and `paper-writer` own final Chinese academic phrasing.
- When a task starts broad but becomes specific during execution, narrow to the minimal stable set rather than dropping a still-needed skill prematurely.

## Minimal-skill policy

- Use the smallest set of skills that fully covers the task.
- Do not invoke `literature-review` for purely local checkpoint analysis.
- Do not invoke `citation-management` when no external reference verification is needed.
- Do not invoke `peer-review` for simple formatting-only writing requests.

## Typical patterns

- `帮我比较 Derf 和 DyT，并写进论文结果分析`
  Use `experiment-runner` + `paper-writer`

- `帮我审查这套实验设计是否合理`
  Use `experiment-runner` + `peer-review`

- `帮我写相关工作并补齐最近文献`
  Use `literature-review` + `paper-writer`

- `帮我核对这些 DOI 并整理参考文献`
  Use `citation-management`

- `帮我把本地实验结果和外部文献一起写成讨论部分`
  Use `experiment-runner` + `literature-review` + `citation-management` + `paper-writer`

- `帮我想实验改进方案，再判断这些改动是否严谨，最后整理成论文里的实验计划`
  Use `experiment-runner` + `peer-review` + `paper-writer`

- `帮我补相关工作，并把这些文献引用格式核对干净`
  Use `literature-review` + `citation-management` + `paper-writer`

- `根据本地 checkpoint 结果，结合外部文献写讨论，并检查结论是不是夸大`
  Use `experiment-runner` + `literature-review` + `peer-review` + `citation-management` + `paper-writer`

## References

- Read `references/routes.md` for quick examples and boundary cases.
