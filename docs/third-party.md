# Included source and provenance

`provenance/sources.json` records the source repository URLs, exact heads and
working-tree status at export. A head alone would miss the supervisor's local
changes. The adjacent patches capture tracked local edits; the source snapshot
also includes untracked runtime files such as the current CaT profile and tests.
`exported-source-hashes.json` describes the initial snapshot before handoff fixes.
`handoff-changes.json` records changes/omissions made to that snapshot.

| Component | License record retained |
| --- | --- |
| STORM | `storm/LICENSE` (MIT, NVIDIA); embedded differentiable robot model license |
| Franka description / ROS / libfranka | Their root Apache-2.0 licenses; libfranka common and bundled test-library licenses |
| AGIMUS common launch utilities | `ros2_ws/src/agimus-demos/agimus_demos_common/LICENSE` |
| Linear feedback controller and messages | Respective root license files |
| STORM ROS bridge / demo | Original package manifests declare BSD-2; existing file headers/metadata retained |

This source distribution preserves the components' individual licensing rather
than assigning a new blanket license. The vendor code remains visible for
traceability; the student's main edits should be in the bridge, STORM collision
path and `research/` examples described in the guide.
