# Experiment source

`src/experiment` owns the paper workflow and `src/integrations` owns concrete
external adapters. The public path is the shell, scheduler, one task runner,
then one native launcher. Python modules are seams behind that path, not public
alternate runners.

Read `docs/ARCHITECTURE.md` before adding a file. New functionality belongs to
the existing owner whose interface already matches it.

