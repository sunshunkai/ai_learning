---
name: release-notes
description: Turn a set of changes into clear release notes grouped by user impact.
triggers: release notes, changelog, version summary, 发布说明, 更新日志
allowed_roles: developer, maintainer
version: 1.0.0
---

# Release Notes

Use this skill when the user asks to summarize changes for a release or
maintain a changelog.

## Steps

1. Separate user-visible changes from internal maintenance.
2. Group entries into Added, Changed, Fixed, Deprecated, and Security.
3. Call out breaking changes and migration steps first.
4. Preserve issue or pull-request references supplied by the user.
5. Do not invent a version number or release date.

## Output

Return Markdown with an `## Unreleased` heading unless the user supplied a
version. Keep each entry concise and action-oriented.
