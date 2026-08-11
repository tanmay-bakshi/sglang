# Common-clock request tracing

`SGLANG_REQUEST_TRACE=1` enables a request-correlated JSON line stream in the
existing gateway, prefill, packed-transfer, and decode service logs. Every line
starts with `SGLANG_REQUEST_TRACE ` and carries schema version 1.

The clock is Linux `CLOCK_MONOTONIC` (`time.monotonic_ns()` in Python and
`clock_gettime(CLOCK_MONOTONIC)` in the gateway). Its epoch is shared by every
process on one machine, but not by different machines. This is deliberately a
domain-local trace contract.

The gateway-created child request UUID is the inference request ID forwarded to
both model services. Scheduler events bind that ID to the decoder-minted
bootstrap room. Packed worker events use the room plus the 16-byte allocation
generation, so room reuse cannot join two request lifetimes. A process-local
sequence resolves equal timestamps and proves log order.

Schema 1 covers these boundaries:

- gateway routing choice, selected prefill and decoder, group, and load snapshot;
- completed prefill batch, token shape, and actual CUDA-graph decision;
- packed transfer begin and end for every source writer, including logical and
  physical geometry;
- decoder handoff token ready for streaming;
- first decode issue with total decoder batch occupancy; and
- first decode result with accepted-token count and graph decision.

Tracing is disabled by default. The disabled Python path is one cached boolean
branch before tuple construction, clock reads, JSON construction, and logging.
The gateway checks the same cached opt-in before cloning or parsing its request
body. An invalid opt-in value fails closed.

## Frozen non-perturbation measurement

Tracing cannot ride an authoritative campaign until this prospective check
passes on the exact 8K stack under test:

1. Run one paired p8192/o256 c8 pass and one paired p8192/o256 c16 pass. Bind
   each pair to the same workload seed and project every row to at least 30
   seconds from the exact incumbent's latest sealed evidence.
2. Both arms use one source revision, gateway binary, launcher, model weights,
   topology, CUDA-graph configuration, kernel menu, and DFlash configuration.
   The only arm difference is `SGLANG_REQUEST_TRACE=0` versus `1` on the
   gateway and every model-service process.
3. Admit tracing only when all of the following hold:
   - raw output throughput regresses by no more than 1%;
   - TTFT p99 increases by no more than 5 ms;
   - TPOT p99 increases by no more than 0.1 ms;
   - zero quality, lifecycle, or identity drift, request failures, or taint;
   - every successful request has the exact event multiplicities implied by
     the declared TP topology, every room joins to one request ID, and there are
     no duplicate begin/end or first-token boundaries; and
   - timestamps are nondecreasing by process sequence and satisfy the causal
     path from route through first token.
4. Seal the raw rows, trace shards, parser output, source identity, environment
   delta, and decision in one hash-bound receipt.

If any gate fails, authoritative runs keep tracing disabled. The same build may
still run a separate attribution row, but its wall-time metrics are not
decision evidence.
