# Integration edit guide

Integrations satisfy existing seams. AndroidWorld owns native I/O and method
construction; AppAgent and MobileGPT adapters translate their external
contracts; B-MoCA adapters preserve official reward semantics; RunLog and
replay adapters convert evidence into the shared runtime.

An adapter may translate, validate, or patch an upstream dependency. It may not
own task scheduling, Function persistence, a second action mapper, or a second
episode lifecycle.

