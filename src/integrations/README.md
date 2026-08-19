# Integration edit guide

Integrations satisfy existing seams. AndroidWorld owns native I/O and method
construction; AppAgent and MobileGPT adapters translate their external
contracts; B-MoCA adapters preserve official reward semantics; RunLog and
replay adapters convert evidence into the shared runtime.

An adapter may translate, validate, or patch an upstream dependency. It may not
own task scheduling, Function persistence, a second action mapper, or a second
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
  -> src/integrations/mobilegpt_memory.py   # memory checks and evidence
  -> bash scripts/exp/test_provider.sh mobilegpt

AppAgent: src/experiment/appagent_source.py
  -> src/integrations/appagent.py           # upstream memory/runtime format
  -> bash scripts/exp/test_provider.sh appagent
```

This is a change path, not a second execution path. The provider source owner
prepares or validates memory; the common AndroidWorld chain consumes the
result. Do not add provider selection to `androidworld.py` or create another
runner.
