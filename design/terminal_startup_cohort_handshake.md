# Terminal Startup Cohort Handshake

## Decision

The terminal owner freezes its producer directory before the runtime starts. A
process therefore needs every authenticated native peer identity before it can
construct `NativeTerminalRuntime`. Request-time decoder registration and
request-time prefill route lookup are too late and cannot mutate that directory.

Each deployment group has one process-lifetime startup registry. Its static
expectation is derived from the canonical
`packed-terminal-deployment-cohort-v1` document, whose exact bytes and SHA-256
are the deployment epoch. The startup protocol does not define a second static
manifest. It enriches that authenticated membership with the identities which
exist only after NIXL initialization:

- exact NIXL process generation;
- exact agent name; and
- SHA-256 of the complete NIXL agent metadata.

The registry admits every source and decoder TP rank in the group, then returns
the same canonical matrix to every waiter. No rank can start its terminal
runtime from a partial population.

## Anchors and current startup wall

`NixlKVManager.__init__` creates `process_generation` and the NIXL agent before
it snapshots `agent_metadata` in `python/sglang/srt/disaggregation/nixl/conn.py`.
That is the earliest point at which a rank can make a complete startup
advertisement.

The source currently publishes its generation, agent name, metadata, and
metadata digest from `NixlKVManager._bootstrap_transport_registration` through
`CommonKVManager.register_to_bootstrap`. The prefill bootstrap listener accepts
that registration in `CommonKVBootstrapServer._handle_route_put`.

Decode currently discovers source identities only when
`NixlKVReceiver._load_bootstrap_peers` consumes request-selected `/route`
responses. Source currently discovers decoder identities only when its ZMQ
bootstrap worker parses a request-era decoder registration and calls
`NixlKVManager._add_remote_peer`. Neither route provides a process-lifetime
bidirectional population before the terminal actors are constructed.

`NativeTerminalRuntime.__init__` builds the complete producer lookup keyed by
producer class and exact authenticated issuer digest, and registers all specs
with the native owner. `NativeTerminalRuntime.start` makes that directory live;
`NativeTerminalRuntime.python_producer_id` only resolves pre-registered
authorities. Dynamic producer registration is deliberately absent.

The production launcher currently uses `model_service_startup_phases`. It
starts all model services concurrently under exclusive placement, but starts
prefill and waits for `/health_generate` before starting decode under shared
placement. That phased path deadlocks once prefill readiness correctly depends
on the complete bidirectional startup matrix.

## Required composition

The canonical per-group manifest has this static shape:

```text
packed-terminal-deployment-cohort-v1
  group_id
  model_fingerprint
  logical_kv_layout_fingerprint
  prefill
    id
    launch_instance_id
    origin
    bootstrap_endpoint
    tensor_parallel_size
  decoders[]
    id
    launch_instance_id
    origin
    tensor_parallel_size
```

The expectation adapter maps `prefill` to the sole source service and each
decoder entry to one decode service. It supplies the document SHA-256 as
`cohort_sha256`; no explicit launch epoch is introduced. `origin` is retained
in each startup expectation and advertisement, so a live process cannot claim
another statically selected service route.

After all long-lived NIXL memory sections have been registered and
`agent_metadata` has been frozen, every source and decode rank sends a
`TerminalStartupRankAdvertisement` to the group's prefill bootstrap endpoint.
The endpoint decodes the bounded canonical bytes and calls the sole
`TerminalStartupCohortRegistry.register_and_wait`. The handler blocks on the
registry condition, not on a polling cadence. Completion returns canonical
`TerminalStartupCohortMatrix` bytes. Any conflict, timeout, replaced rank,
duplicate generation, duplicate agent identity, or explicit failure wakes all
waiters into one sticky epoch failure.

Each rank decodes the returned matrix, proves it against its independently
loaded static expectation, and verifies its own matrix row against its local
process generation, agent name, metadata digest, TP rank, and TP width. It then
derives a least-privilege Python producer plan before constructing its sole
terminal runtime:

- one local producer and one local receipt producer;
- control and receipt producers for every cross-role rank;
- receipt producers for same-service TP peers; and
- no same-role authority for unrelated decode replicas.

Native NIXL/CUDA producers append their specs starting at the returned
`next_producer_id`. The complete tuple is passed to
`NativeTerminalRuntime.__init__` once and is never mutated.

The existing full-metadata routes remain responsible for native peer creation.
They must revalidate service identity, process generation, agent name, and
metadata digest against the sealed matrix before calling `_add_remote_peer`,
`_load_bootstrap_peers`, or dispatching packed control. A matrix generation is
not, by itself, route authentication.

## Launcher order

The launcher must separate listener liveness from generation readiness:

1. start the prefill model service and its bootstrap listener;
2. wait only for bootstrap listener liveness, not `/health_generate`;
3. start every decoder in the group;
4. allow all ranks to complete the bidirectional startup join; and
5. wait for normal generation readiness on every model service.

Exclusive placement may continue to start every model service concurrently.
Shared placement needs the listener-first order above. If colocation makes
source model loading retain too much memory for decoder startup, the placement
contract itself is incompatible with an immutable process-lifetime identity
cohort; waiting for source generation readiness is not a valid workaround.

## Failure and restart semantics

The group SHA-256 plus observed matrix digest identify one exact startup epoch.
An identical advertisement can retry idempotently after sealing. A changed
generation or any other changed row fails the group. Recovery requires fresh
launcher launch instance IDs, a fresh canonical cohort document, replacement
registries, and replacement terminal runtimes. There is no mutable
registration path.

The join deadline begins at the first accepted registration, not bootstrap
listener creation, because model initialization can legitimately precede the
join. A registration arriving at or after the deadline fails before it can
complete the population, so sealing is not scheduler-race dependent.

## Integration boundary

The tested registry, canonical wire codecs, and producer-plan derivation are
landed independently. The remaining integration deliberately composes in the
canonical manifest and server configuration lane because the bootstrap server
cannot authenticate a startup advertisement without its exact group
expectation. Half-wiring a public handler without that static input would turn
the listener into an identity oracle and is rejected.

The composition lane owns:

- canonical-manifest-to-expectation adaptation;
- local service selection and startup timeout injection;
- bootstrap server registry construction and the blocking startup route;
- decode registration to the source-owned registry;
- local row verification and complete producer-spec construction;
- full-metadata route revalidation against the sealed matrix; and
- listener-first launcher sequencing.
