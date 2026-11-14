# Localized door counting

The interactive door-count path requires an object inventory before accepting a
total. A wing-level estimate such as rooms multiplied by a typical door layout is
not counting evidence. Other question types retain their existing tools.

## Delegation and frames

- Gemini selects regions or registered room IDs. Door algorithm requests also
  return room navigation once, without another algorithm run. Pending algorithms
  still do not block visual inspection.
- Each native tile has an explicit local frame and original-page bounds. Generic
  tasks receive a coordinate conversion; counting workers receive the original
  user question instead of leading coordinator assumptions or repeated-room math.
- Workers inspect only visible symbols within their tile, not an unseen whole wing.
  Door roles still require adjacent space context. Double-leaf pairs count as one
  opening unless the user explicitly requests leaves.
- Each worker returns `objects`: local normalized box, image ID, counting unit,
  supported/uncertain status and a short visible cue. `scope_complete` refers only
  to its tile. Missing inventories never mean zero. Dense count responses have a
  6144-token output allowance (normal specialist requests remain 2048).

## Host inventory and final answer

`count_evidence.py` transforms boxes to the original frame. Stable IDs tie each
candidate to its source crop. High-IoU duplicates are merged conservatively;
conflicting statuses or remaining overlapping candidates stay uncertain. A
complete smaller reinspection can replace an old incomplete region. The object
inventory survives provider recovery and is retained on an abstention.

Gemini submits `count_basis: {scope: "complete"|"partial", unit: "door_opening"|
"door_leaf"}`. Optional `object_ids` may select an explicit subset; omission uses
all supported inventory objects. A complete total is accepted only when the value
matches the unique supported inventory, there are no uncertain objects or
unresolved requested tiles, and every retained tile declares complete inspection.
Gemini remains responsible for selecting the entire relevant drawing scope.

Unsupported totals become an incomplete response with the available localized
candidates. The host writes a count briefing from that inventory so unsupported
aggregate prose cannot be described as confirmed. It attaches individual source
citations, with up to100 inline source references; all objects remain in the
separate inventory. Multi-page source projection applies to both representations.

These checks verify bookkeeping and source bounds, **not physical truth**. Model
false positives, missed doors, role mistakes and a missed whole region remain
possible. Geometry deduplication is conservative and can require closer review.
English/German door-count phrasing activates the guard; other types/languages
remain on the general visual-evidence path. This is not a general verified counter.

## Scheduling and presentation

Oversized selections dispatch in the requested priority order up to the remaining
call/crop allowance. Every omitted tile is explicitly deferred with its bounds and
reason; feasible work is no longer rejected as a batch. Successful tiles are reused;
failed/incomplete tiles allow one retry. Argument repair does not consume the12
productive tool slots, while18total attempts, turn limits and the shared deadline
still bound execution. Call/resolution/concurrency/time budgets are unchanged.

The drawing shows solid green model-supported boxes and dashed amber uncertain
boxes. A compact toolbar toggle controls them; the answer gives provisional counts
and explains that neither status is human verification. Boxes survive refresh and
are filtered by source identity, page and bounds. `completion_reason` distinguishes
`inspection_incomplete` from `source_insufficient`; interrupted processing should
not ask the user to supply a clearer drawing by default.

## Verification

Scripted Python regressions cover coordinate contracts, room context, localized
IDs, unsupported totals, duplicate/uncertain candidates, selective retries,
recovery, page projection and partial dispatch. Browser regressions cover both
candidate states, toggling, refresh persistence and wrong-source rejection.
These tests use simulated providers and do not establish live accuracy or latency.
