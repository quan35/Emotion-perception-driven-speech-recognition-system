# Section Patterns

This reference contains concise writing patterns for common paper sections in this repository.

## Related Work pattern

Start from broad technical evolution, then narrow to the precise gap addressed here. Prefer thematic grouping over chronological dumping. A strong closing move is to state that prior work either emphasizes single-task SER, single-language settings, or alternative encoder families, whereas this repository focuses on a shared Whisper-based ASR and SER pipeline under a fixed six-class emotion space.

## Methods pattern

Explain the problem formulation, then the architecture, then the data protocol, then optimization and monitoring. If the text mentions the early `CNN+BiLSTM+Attention` route, frame it as historical exploration rather than the current default system.

## Result analysis pattern

Use a three-step structure:

1. State the observed metric change.
2. Interpret the likely technical reason with restraint.
3. State the practical implication or limitation.

Useful sentence skeleton:

`与基线配置相比，当前设置在 <metric> 上取得了 <direction> 的变化。这表明 <technical interpretation>. 然而，该改动在 <other metric> 上并未同步改善，因此更适合被表述为一种具有条件优势的配置，而非无条件优于基线的方案。`

## Figure caption pattern

Captions should identify:

- what is being visualized
- the comparison scope
- the most important trend

Avoid interpreting beyond what the figure directly supports.

## Evidence boundary

- Local numerical claims should come from checked artifacts.
- Literature claims should come from verified sources.
- Never merge local evidence and external evidence into a single unsupported sentence.
