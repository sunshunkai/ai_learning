---
name: json-normalizer
description: Normalize inconsistent JSON into a stable, validated structure.
triggers: json, normalize json, format json, 格式化json, 规范化json
allowed_roles: developer, analyst
version: 1.0.0
---

# JSON Normalizer

Use this skill when JSON is malformed, contains inconsistent keys, or needs a
stable output shape.

## Steps

1. Parse the input before attempting any rewrite.
2. Identify required fields, optional fields, and incompatible value types.
3. Normalize names and types consistently across all records.
4. Preserve unknown fields unless the user explicitly asks to drop them.
5. Report validation failures separately from normalized output.

## Output

Return valid JSON in a fenced code block and list any normalization rules that
were applied. Ask one short clarifying question when the target schema is
ambiguous.
