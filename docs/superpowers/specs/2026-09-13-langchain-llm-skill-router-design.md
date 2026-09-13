# LangChain LLM Skill Router Design

## Goal

Add a LangChain example that uses structured model output to select Skills when
deterministic routing rules do not match, while preserving the existing
rule-based and Agent-tool paths.

## Context

The existing LangChain example already exposes a lightweight Skill catalog to
the main Agent and lets the model call `search_skills` and `read_skill`.
Separately, `tools/skill_session.py` defines `HybridSkillRouter`, but the
LangChain example does not provide a LangChain-native fallback router.

The new example demonstrates a two-stage design:

1. Deterministic rules handle explicit selections and known triggers quickly.
2. A structured LangChain model call handles semantic cases that the rules miss.

The main Agent tool flow remains available and is not replaced.

## Architecture

Create `langchain-demo/llm_skill_router.py` with:

- `SkillSelection`: a Pydantic model containing `skill_names` and `reason`.
- `LangChainSkillRouter`: a `SkillRouter` implementation that:
  - returns an empty, auditable `RouteDecision` when no candidates exist;
  - sends only candidate names and descriptions to the routing model;
  - uses `model.with_structured_output(SkillSelection)`;
  - filters model-returned names against the candidate set;
  - deduplicates names while preserving model order;
  - returns `RouteDecision(..., strategy="model")`.

Modify `langchain-demo/16_skill_loading.py` to:

- add `--router {rule,hybrid}`, defaulting to `hybrid`;
- keep `--inspect` network-free and rule-based;
- build the routing model only for non-inspect execution;
- use `HybridSkillRouter` with `RuleBasedSkillRouter` as primary and
  `LangChainSkillRouter` as fallback when `--router hybrid` is selected;
- call `session.route(prompt)` before constructing each Agent so model-selected
  Skills are actually preloaded and can be inspected in the Agent transcript;
- print the pre-routing decision before the Agent response.

The implementation uses the existing `SkillSession` authorization, deduplication,
and context-budget checks. The model router never loads content directly.

## Data Flow

```text
user prompt
  -> RuleBasedSkillRouter.select()
  -> explicit or trigger match?
       yes -> RouteDecision(strategy="explicit" | "rule")
       no  -> LangChainSkillRouter.select()
                -> model.with_structured_output(SkillSelection)
                -> filter to candidate names
                -> RouteDecision(strategy="model")
  -> SkillSession.route() validates names and calls load()
  -> render() includes loaded SKILL.md bodies
  -> create_agent() continues with tools available for later Skills
```

## CLI Behavior

`--router rule` preserves the current deterministic behavior. A rule miss
leaves the Skill context unchanged, and the main Agent may still use its tools
to discover a Skill.

`--router hybrid` adds a separate structured model call only when the rules
return no selection and no explicit Skill decision. A matched trigger or an
explicit `@skill:name` does not call the routing model.

`--inspect` remains usable without API credentials. It prints the selected
router mode and a rule-based preview without making a model call.

## Error Handling

- No candidates: return an empty model decision without calling the model.
- Unknown model-returned names: discard them.
- Duplicate model-returned names: keep the first occurrence.
- All returned names invalid: return an empty model decision with a reason.
- Model invocation exceptions: propagate to the existing top-level error
  handler in `16_skill_loading.py`.
- Invalid names that bypass the router: remain protected by
  `SkillSession.route()` validation.

## Testing

Add focused tests using a fake structured-output model:

- no candidates does not invoke the model;
- candidate names and descriptions are included in the routing prompt;
- valid names are returned in order and use `strategy="model"`;
- unknown and duplicate names are removed;
- `HybridSkillRouter` invokes the LangChain fallback only on a rule miss;
- explicit loaded Skills do not invoke the fallback.

Run the complete existing test suite after the focused tests pass.

## Documentation

Update `langchain-demo/README.md` and `langchain-demo/KNOWLEDGE_GUIDE.md` with:

- the difference between main Agent tool selection and independent LLM
  pre-routing;
- the `--router rule` and `--router hybrid` commands;
- the fact that hybrid routing makes an additional model call only on a rule
  miss;
- the trade-off between deterministic latency/cost and semantic flexibility.

## Non-Goals

- Replacing Agent tool selection with pre-routing.
- Adding retrieval embeddings or a vector database.
- Calling the model during `--inspect`.
- Loading full Skill bodies during route selection.
- Changing the Claude SDK or Anthropic container mode.
