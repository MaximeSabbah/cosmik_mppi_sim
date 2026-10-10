# Adding human-motion prediction

The first research milestone is a working time-indexed obstacle path with a
zero-order-hold predictor. Then compare it against a simple constant-velocity
predictor, before adding a learned or articulated model.

## Current path through the code

1. COSMIK-format `visualization_msgs/MarkerArray` arrives at
   `/rtcosmik/collision_markers`. Cylinders have centre pose, diameter in `scale.x/y`,
   height in `scale.z`, and their axis along local Z. Positions/scales are metres.
   The included fixture publishes one cylinder at 30 Hz on a 12-s periodic path.
2. In `storm_mppi_planner_node.py`, `_collision_markers_callback` stores the latest
   array; `_collect_dynamic_collision_observations` transforms its geometry to
   `fer_link0`.
3. `_initialize_dynamic_collision_slots` fixes the slot count and initial dimensions.
   `_update_dynamic_world_spheres_from_observations` creates five spheres per
   cylinder by default. Observations are currently assigned by list order, and
   stable marker IDs and observation timestamps are not carried into this representation.
4. The bridge sends `update_params(dynamic_world_spheres=...)` through the existing
   STORM task/optimization process. `ArmBase.update_params` passes it to
   `PrimitiveCollisionCost.set_dynamic_world_spheres`.
5. That setter accepts **[N,4]**, where each row is `[x,y,z,r]`. Dynamic collision
   evaluation broadcasts those same spheres across all particles and horizon steps.
   The collision violation is supplied to the existing CaT weighting in `mppi.py`.

This reacts to observed motion as new frames arrive. It does not predict human
motion. `exact_latency_prediction` predicts robot state during command latency;
it is a different mechanism.

## Proposed research boundary

`research/capsule_prediction.py` contains a small, tested reference contract:

```python
Observation(stamp, frame, ids, endpoints, radii)
# endpoints: [capsules, 2, 3], radii: [capsules]

forecast = predict(current, rollout_times, previous=None)  # hold
forecast = predict(current, rollout_times, previous=old)   # constant translation velocity
# forecast.endpoints: [horizon, capsules, 2, 3]
```

Each ID should derive from `(marker.ns, marker.id)`. Store observations in a
bounded history keyed by ID, with timestamps and a single explicit coordinate
frame. The CV example matches IDs even if an array is reordered. A new ID starts
with zero velocity. It holds length and orientation fixed; it is a comparison
baseline, not a full human-motion model.

The cylinder-to-capsule convention must be explicit: the current bridge places
sphere centres along `max(height - 2*radius, 0)`, preserving the marker's total
extent convention. Do not use surface endpoints as centre-line endpoints and
then add the radius again. Match the existing geometry exactly for the hold test.

`rollout_times` must be the actual absolute evaluation times of robot states,
expressed in the observation clock domain. Inspect the dynamics model's
`_dt_h` and `_traj_tstep` in `urdf_kinematic_model.py`, and the asynchronous command
timing in `mpc_process_wrapper.py`. Include observation age and robot rollout
start/latency consistently. Do not assume that callback receipt time is capture
time or that the horizon will always use uniform 0.02-s steps.

The next implementation should carry `[H,N,4]` future spheres plus timing metadata
through the process boundary, preserving horizon index when collision evaluation
flattens particle and horizon dimensions. Both the ordinary and fused dynamic
collision paths need explicit handling. Keep `[N,4]` as the baseline mode and
preserve static world collisions and the existing CaT violation/reduction rules.
Multiplying stationary obstacles in number, or changing only RViz markers, does
not implement future human collision evaluation.

Do not connect the standalone `predict` function to just the ROS visualization
and label the controller predictive: the optimizer must consume the forecast.

## Suggested sequence and acceptance checks

1. Reproduce both smoke tests and save `doctor.sh` output. Make a baseline Git tag.
2. Add stable-ID history and visualization of forecasts. Test reordered, missing,
   duplicated, stale and out-of-order observations; reset history on a clock reset.
3. Add time-indexed geometry transport and collision evaluation. A stationary
   forecast repeated over H must match the original costs, CaT violations and
   weights within numerical tolerance. Test H=1 and multiple particles to expose
   flattening/index mistakes.
4. Use a crossing-obstacle fixture where the future intersection differs from
   the current geometry. Check a hand-computed clearance/violation at each time.
   Verify the optimizer actually receives new predictions through its worker.
5. Compare hold and CV using identical targets, human trajectories, tuning and
   seeds. Record arrival success, timeouts, minimum sphere/capsule surface distance,
   joint speed/acceleration, optimizer rate, observation age and prediction latency.
6. Add the research predictor behind an explicit mode switch, including its
   missing-data fallback and uncertainty policy. Keep hold as the regression mode.

The current transport has no complete stale-human policy: cached markers can
remain in the controller, and dynamic slots are initialized from the first useful
frame. Marker lifetimes and DELETE actions are not a complete tracking protocol.
Treat changes in IDs/count/radii, occlusion, frame transforms and restart behavior
as part of the prediction integration, not hidden assumptions.

## Live input later

Run `./scripts/run.sh synthetic_human:=false collision_topic:=/rtcosmik/collision_markers`.
For a publisher on another host/domain, set `DEMO_DOMAIN_ID` and
`DEMO_LOCALHOST_ONLY=0` for all relevant terminals. Keep this simulation on a domain
separate from an active hardware controller. `world` and `fer_link0` are coincident.
For research, align live capture timestamps to simulation time before estimating
velocities or horizons. No old rosbag is required for the default demonstration.
