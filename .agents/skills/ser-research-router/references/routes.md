# Routing Notes

This reference gives compact boundary cases for the router skill.

## Experiment-first tasks

If the request mentions local files, checkpoints, `config.yaml`, `summary.json`, `history.npz`, `Derf`, `DyT`, `freeze_strategy`, `training_mode`, `subset_epoch_caps`, or `best_shared_model`, or the user says `实验改进`, `实验优化`, `消融`, `对比实验`, `跑一下`, `复盘这次实验`, start from local evidence and favor `experiment-runner`.

## Writing-first tasks

If the request asks for `改写`, `润色`, `扩写`, `写成论文语言`, `写成中文学术表述`, `补一段结果分析`, `补方法部分`, `写图表说明`, `写讨论`, or `写摘要`, favor `paper-writer`. Add other skills only when the writing depends on external evidence or local metrics.

## Literature-first tasks

If the request asks for `综述`, `相关工作`, `研究现状`, `研究空白`, `state of the art`, `最新工作`, `补文献`, or `找近几年工作`, favor `literature-review`.

## Citation-first tasks

If the request asks for `DOI`, `BibTeX`, `参考文献`, `引文核对`, `作者年份页码`, `引用格式`, `补齐元数据`, or duplicate checks, favor `citation-management`.

## Review-first tasks

If the request asks whether a design is `合理`, `公平`, `严谨`, `可复现`, `是否夸大结论`, `是否站得住脚`, or `审稿人会不会质疑`, add `peer-review`.

## Mixed-evidence tasks

If the request simultaneously mentions local experiment outputs and external papers or references, keep `experiment-runner` active for local evidence and add `literature-review` or `citation-management` rather than replacing the local skill.

If the request asks to `先分析实验，再写成论文`, combine `experiment-runner` with `paper-writer`.

If the request asks to `一边补文献一边写讨论`, combine `literature-review` with `paper-writer`, and add `citation-management` when references must be verified.

If the request asks to `先审查设计是否合理，再给出修改建议并写成论文表达`, combine `experiment-runner`, `peer-review`, and `paper-writer`.
