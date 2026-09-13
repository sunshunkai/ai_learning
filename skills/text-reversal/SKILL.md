---
name: text-reversal
description: Reverse user-provided text while preserving case, punctuation, and whitespace.
---

# Text Reversal

Use this skill when the user asks to reverse text, a word, or every line of a
text block.

## Steps

1. Identify the exact text the user wants reversed.
2. Remove only the surrounding whitespace from each input line.
3. Reverse the characters in each line.
4. Preserve original casing, punctuation, numbers, and non-ASCII characters.
5. Put the result in one fenced code block.
6. Say that the `text-reversal` skill was used.

## Example

Input:

```text
AI Learning
```

Output:

```text
gninraeL IA
```

If the input is ambiguous, ask one short clarifying question instead of
guessing.
