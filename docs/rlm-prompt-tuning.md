# RLM Prompt Tuning Guide

This guide is for teams customizing RLM prompts in RLM Studio.

The main rule is simple: prompt changes must respect the actual tool contract. If a prompt teaches an impossible workflow, better wording will not save it.

## Recommended Defaults

- Keep the built-in v2.0 base prompt as the protocol layer.
- Treat profile prompt text as a supplement, not a replacement.
- For multi-document runs, read the document index first.
- Prefer direct jumps via `content_start` offsets from the index.
- Use `grep(..., context_lines=...)` when you need local context around a marker or keyword.
- If coverage is incomplete, instruct the model to say so explicitly.

## Safe Multi-Document Pattern

Use this mental model:

1. `peek(0, 1200)` to read the document index.
2. Use `content_start` from the index to jump into a relevant file.
3. Use `grep()` when you need to find a section or term inside that file space.
4. If `grep()` is used, rely on the returned `char_offset` for follow-up `peek()` calls.
5. Finalize once the answer is supported, and state any coverage gaps.

## Anti-Patterns

Do not put these into custom prompts:

- "Use grep, then peek from the returned position" unless grep actually returns a character offset.
- "You must inspect every file before finalizing" without a budget escape hatch.
- "Use your full step budget" or other instructions that fight convergence controls.
- Prompt text that restates the full JSON protocol differently from the base prompt.
- Model-specific heuristics presented as universal rules.

## Good Supplement Characteristics

A good profile supplement should:

- narrow behavior for a workload, not redefine the protocol
- tell the model how to handle incomplete evidence
- bias toward supported tools and observable outputs
- be short enough that it does not crowd out execution history

## Model-Specific Guidance

For smaller or local models:

- keep supplements shorter
- avoid rigid multi-step recipes
- prefer direct offset-based navigation over inferred navigation
- use lower step budgets unless the model has already shown stable loop behavior

For stronger frontier models:

- encourage per-document summaries before synthesis
- allow deeper inspection when the query truly spans multiple files
- still require explicit uncertainty when evidence is incomplete

## Validation Checklist

Before shipping a prompt change, run these scenarios:

- Single-document summary
- Targeted question about file 2 in a multi-file upload
- Cross-document comparison where both files matter
- Small-model malformed JSON case
- Budget-limited run where full coverage is impossible

The change is not ready if any of those scenarios regress materially in answer quality, step count, or completion behavior.

## Example Supplement

```text
Additional guidance for this profile:
- When the document index lists multiple files, read it first and use each file's content_start offset to jump directly into relevant files.
- Use grep(..., context_lines=...) when you need local context around a marker or keyword; grep returns char_offset values for precise follow-up peeks.
- For cross-document questions, build a short per-document summary before synthesizing.
- If not all relevant files were inspected within budget, say which files were covered and lower confidence accordingly.
```
