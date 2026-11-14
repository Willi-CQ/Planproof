# Quiet drawing workspace

Branch: `design/premium-agent`, based on the working MVP at `bf64999`.
Main and its running UI remain separate. The backend, agent prompts, routing,
models, budgets, recovery and algorithms are unchanged.

Design workflow: Meng To's [Design-First UI Prompting skill](https://github.com/MengTo/Skills/blob/main/agent-skills/ui/design-first-ui-prompting/SKILL.md),
read on 2026-11-14. Applied as an implementation spec, not an image-generation
prompt. The reference is an Apple-like native utility: quiet, precise, direct.

- Outcome: upload one drawing, ask Planproof, inspect the cited source.
- Layout: focused upload screen; then an expanded drawing canvas beside a wide,
  full-height floating conversation panel. On phones, two tabs keep both usable.
  Page navigation appears only in the bottom drawing toolbar, beside zoom controls.
- Typography: locally available system fonts, restrained weight, readable body
  text. No external font fetches.
- Materials: near-white background, white panels, subtle borders, soft shadows.
  Deep green (#001D00), observed on [Stoa](https://stoabuild.com/) on 2026-11-14,
  accents actions and evidence. Supporting sage shades are our own derived palette.
  Larger chat text and generous spacing keep the conversation easy to read.
- Hierarchy: a single primary action per state. Tool counters, source dimensions,
  model names and connection diagnostics are absent from the normal flow.
- Behavior: drop/upload, keyboard send, native-pixel highlights, zoom and pan.
  Suggested questions only fill the composer; they never trigger a model call.
- Sending a question allows Planproof's AI services, disclosed next to the
  composer. The existing request permissions are set together on that action.
  Upload alone never calls inference. No provider chooser or automatic retry.
- Details: relevant uncertainty remains available under “About this answer”.
  Stateless question scope is explained in optional help, without changing the agent.
- Safety: no fake answers, animations implying unreported tool execution,
  invented confidence scores or altered drawing pixels.

Verify source alignment, desktop/mobile layouts, failure recovery and keyboard
access. Honor reduced motion. Screenshots and private drawing checks stay under
the ignored `.artifacts/` directory.
