# AndroidWorld Paper Experiment

This context defines the reproducible AndroidWorld experiment vocabulary shared
by every evaluated method.

## Language

**AndroidWorld Experiment Environment**:
The single owner of the official environment and episode lifecycle shared by
every evaluated method.
_Avoid_: Method environment, baseline environment

**Method Adapter**:
The method-specific bridge that constructs one evaluated method for the shared
AndroidWorld Experiment Environment without owning its lifecycle.
_Avoid_: Method runner, method environment

**Source RunLog**:
One complete successful AndroidWorld RunLog used as the only evidence for
preparing a baseline input.
_Avoid_: source asset, source pool, teacher data

**Prepared Memory**:
One provider-owned input bundle compiled from a Source RunLog for AppAgent,
MobileGPT, or another external baseline.
_Avoid_: Function, catalog, replay script

**Run Check**:
A read-only check that decides whether dependencies, evidence, device state,
and Prepared Memory are ready for one run.
_Avoid_: preflight layer, validation service

**Local Index**:
The single `data/current.json` record that tells runtime which already-sealed
evidence and result records to read.
_Avoid_: catalog, registry, snapshot
