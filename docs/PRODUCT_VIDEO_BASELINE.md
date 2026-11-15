# Product video baseline

The current product interface is frozen for the recording as of 15 November
2026. The recording checkpoint is the Git tag `demo-video-2026-11-15` on `main`.

- Product implementation before this documentation: `74a166a`.
- Exact frontend tree: `77ec18603f3ec800b42a5bcd556a9e412689fd5b`.
- Preserve the current layout, styling, controls, drawing overlays, answer
  presentation, activity display and downloadable report presentation.
- Further work should optimize the agent and backend while preserving frontend
  API contracts and existing saved drawing and conversation compatibility.
- Visual changes require a new explicit request from the project owner. Do not
  combine interface redesigns with agent improvements.

This checkpoint preserves the product for filming; it is not a claim that every
answer is correct. See [the production audit](PRODUCTION_COHORT_AUDIT.md) for
measured results, repairs and remaining validation limits.

To inspect the recording source independently of later work:

```sh
git fetch origin --tags
git worktree add --detach ../planproof-video demo-video-2026-11-15
```

Runtime configuration, credentials, uploaded drawings and local histories are
not stored in this tag. Follow the existing setup documentation to run a fresh
checkout; preserve local runtime data when moving between versions.
