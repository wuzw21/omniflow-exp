# Integration edit guide

Integrations satisfy existing seams. AndroidWorld owns native I/O and method
construction; B-MoCA adapters preserve official reward semantics; RunLog and
replay adapters convert evidence into the shared runtime. External MobileGPT
and AppAgent code is started from its own checkout; OmniFlow only writes their
native warm-start files.

An adapter may translate a source artifact into an upstream file format. It may
not patch upstream prompts/actions, own task scheduling, or create a second
episode lifecycle.

`mobilegpt.py` owns RunLog conversion and the native memory reader.
`mobilegpt_memory.py` owns MobileGPT memory inventory, statistics, manifest and
evidence checks, and prepared-memory validation. Keep those provider contracts
out of the experiment scheduler and AndroidWorld result runner.

## Provider change map

Start with the matching source owner and use the shared harness:

```text
MobileGPT: src/experiment/mobilegpt_source.py
  -> src/integrations/mobilegpt.py          # upstream memory format
  -> src/integrations/mobilegpt_format.py   # official XML encoder only
  -> src/integrations/mobilegpt_memory.py   # memory checks and evidence
  -> bash scripts/exp/test_provider.sh mobilegpt

AppAgent: src/experiment/appagent_source.py
  -> src/integrations/appagent.py           # upstream memory/runtime format
  -> bash scripts/exp/test_provider.sh appagent
```

This is a change path, not a second execution path. The provider source owner
prepares or validates memory; the common AndroidWorld chain consumes the
result. Do not add provider selection to `run_task.py` or create another
runner.
