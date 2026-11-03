# Benchmark v2 — pre-registration

Status: **written 2026-11-02, before any v2 code, data or result exists.** Changes to this plan
after v2 results are seen must be logged in the deviation log of `experiment_protocol.md`.

## Why

Across Phases 4–7 on benchmark v1:
- 76–79% of learned-policy failures end inside junctions;
- red-light violations make up about 40% of failure events (model B).

The rendered frames show two simulator stylisations behind this, both flagged as limitations
in Phase 2:
- junctions are drawn as large unmarked discs (about 29 m across), so the camera sees uniform
  asphalt with no lane or kerb cues;
- signal heads are about 3×5 px at 160×96 and placed only on the far side of the junction.

Phase 8 tests whether acting on introspection improves safety. With a policy that fails in
about 90% of episodes, that test cannot separate the value of introspection from "drive more
slowly". v2 removes the two stylisations. It is justified by realism, not by the learned
models' scores.

## What changes (complete list)

All changes are map/rendering presets. The default configuration stays v1, and v1 maps and
renders are guaranteed bit-identical by a regression test.

1. **Junction geometry.** Junctions are the union of the incident road surfaces, extended to
   the node centre, plus a 45° kerb chamfer between angularly adjacent roads (chamfer leg
   6 m beyond the road edge). This replaces the 14.4 m-radius disc. Sidewalks follow the new
   kerb line.
2. **Guide markings through junctions** (white dashed, 0.15 m):
   - straight-through movements: the lane's left edge, i.e. the road centre line continued
     through the junction;
   - left turns: the turning path's left edge;
   - right turns: none (the kerb guides).

   At two-road bends, the yellow dashed centre line and the solid white outer edge line
   continue through the bend, as on any curved road.
3. **Signal visibility.**
   - Signal heads become 1.4 m wide × 2.6 m tall, up from 1.0 × 1.8.
   - A second, near-side head per approach is added on a kerb-side pole, 8 m past the stop
     line and 1 m outside the road edge, centred 3.6 m above the ground and facing approaching
     traffic. It is visible while waiting at the stop line.
   - Signal timing and logic are unchanged.

Nothing else changes: vehicle dynamics, traffic, pedestrians, events, camera, routes, splits,
seeds, models, training recipes, evaluation protocol and analysis scripts are all identical.

## Checks that gate v2 (simulator only, not the learned models)

- Every lane is drivable including ±1 m to either side (existing test, run on v2 maps).
- Connector turn radii are unchanged; lanes are generated independently of rendering.
- The v1 regression test passes: v1 map rasters and a reference render are bit-identical.
- **Expert on v2** (maps 200–209, 100 episodes, as in Phase 3) completes ≥ 98% of routes with
  zero collisions, red-light violations, off-road events and lane departures.

There is deliberately **no target for the learned models.** Whatever models A–C achieve on v2
is reported as is, next to v1.

## What is re-run on v2

The same pipeline, protocol and seeds as v1:
- expert dataset `expert_v2` (same split map seeds);
- model A × 3;
- model B (H=4) and the B(H=1) control × 3;
- closed loop on the 150 test_id routes (expert, A, B(H=1), B);
- introspection (Phase 6 pipeline);
- uncertainty (Phase 7 pipeline, ensembles retrained).

## Reporting

- All v1 results stay in `docs/results.md` unchanged. v2 results go in new sections with a
  v1-vs-v2 comparison.
- The effect of the visual changes on behaviour cloning is itself reported as a result.
- Phase 8 then runs on v2, provided the expert gate passes. If it does not, v2 is abandoned and
  this is reported.
