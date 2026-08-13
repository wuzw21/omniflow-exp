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
