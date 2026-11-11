# Remote UI Representation Transport

This microbenchmark measures the communication cost of sending an encoded current
UI state to a remote Function Store. It isolates transport from local UI encoding,
Function Recall, and action Transfer computation.

## Protocol

- Remote host: SSH alias `4090`.
- Transport: one persistent encrypted SSH tunnel to an in-memory HTTP receiver.
- Sampling: 30 interleaved requests for each payload size.
- Receiver: reads each payload, verifies its SHA-256 digest, returns a small JSON
  response, and discards the request without writing files or request logs.
- Payload sizes: 1 B control, 4 KiB page vector, 55,197 B FP16 Transfer
  representation, and 106,925 B FP32 Transfer representation.

The two Transfer representation sizes were measured from the same real
OmniTransfer V10 encoded B-MoCA calculator page used by the cloud representation
prototype. The committed benchmark generates deterministic payloads with identical
sizes because transport cost is independent of their encoded values.

## Result

| Payload | Size | P50 | P95 | Paired P50 increment |
|---|---:|---:|---:|---:|
| Control | 1 B | 293.68 ms | 355.48 ms | -- |
| Page vector | 4 KiB | 287.57 ms | 395.96 ms | -0.59 ms |
| FP16 Transfer representation | 53.9 KiB | 300.01 ms | 370.24 ms | 13.27 ms |
| FP32 Transfer representation | 104.4 KiB | 323.48 ms | 385.49 ms | 52.35 ms |

The remote link latency dominates the request. Relative to the one-byte control,
the FP16 Transfer representation adds 13.27 ms at the median. Reducing the request
to a 4 KiB page vector therefore provides little communication benefit on this
link, while that vector alone is insufficient for step-level action Transfer.
Because the public link is noisy, independently aggregated P50 values can cross;
the within-cycle paired increment is the primary payload-cost measurement. The
negative 4 KiB increment is within this link noise and is treated as no measurable
payload overhead.

These numbers are communication-only measurements. They do not include the local
encoder or remote OmniTransfer inference and should not be reported as complete
Function execution latency.

## Reproduction

```bash
./.venv/bin/python tools/measure_remote_transport.py \
  --host 4090 \
  --output /tmp/omniflow-4090-transport.json
```

The script creates no remote files. Its receiver is bounded by a five-minute
timeout and is stopped after the final response.
