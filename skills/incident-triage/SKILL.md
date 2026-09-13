---
name: incident-triage
description: Triage application errors, failures, and stack traces into likely causes and next checks.
triggers: error, exception, stack trace, failure, 报错, 异常, 故障
allowed_roles: developer, operator
version: 1.0.0
---

# Incident Triage

Use this skill when the user provides an error, exception, stack trace, failed
deployment, or production incident and wants diagnosis.

## Steps

1. Extract the observed symptom, first failing boundary, and affected scope.
2. Separate confirmed facts from hypotheses.
3. Rank likely causes by evidence rather than by confidence alone.
4. Propose the smallest diagnostics that can confirm or reject each cause.
5. Identify immediate mitigation separately from root-cause repair.

## Output

Return:

- observed facts
- ranked hypotheses
- next diagnostic commands or checks
- immediate mitigation
- missing information that would change the diagnosis

Read `references/checklist.md` only when the incident spans multiple services
or the initial evidence is ambiguous.
