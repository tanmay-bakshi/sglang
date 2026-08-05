# Gemma 4 component-aware asymmetric-TP staging contract

## Status and scope

This document defines the staging architecture for non-MLA, page-indexed MHA
components transferred between different attention tensor-parallel widths.
The first implementation target is Gemma 4 with PP=1, CP=1, NHD VRAM KV
storage, equal page sizes, and non-replicated contiguous TP partitions.

The existing direct TP-sharded SWA transfer remains the correctness reference.
The existing head-specific staging path is not a safe base for extension. It
infers destination pages from sequence offsets, counts duplicate notifications
as writers, and can reuse source scratch while NIXL still reads it.

Gemma 4 has two independently indexed components:

| Component | Layers | Global KV geometry | Tokens transferred |
| --- | ---: | --- | ---: |
| Full attention | 10 | 4 heads x 512 dimensions | Entire uncached prefix |
| SWA | 50 | 16 heads x 256 dimensions | Final `min(input_tokens, 1,023)` logical prior-KV rows, rounded to complete pages |

Hugging Face's `sliding_window=1024` is inclusive of the current token.
SGLang attention accepts an exclusive prior-KV extent, so
`gemma4_causal.py:get_attention_sliding_window_size` returns
`config.sliding_window - 1`, or 1,023. Define:

```text
swa_logical_tokens  = min(input_tokens, 1,023)
swa_physical_tokens = ceil(swa_logical_tokens / page_size) * page_size
```

The physical count is allocation and copy capacity. Padding rows in its final
page are never logical attention history. With one-byte FP8 KV storage and page
size 1, logical and physical bytes are both:

```text
40,960 * input_tokens + 409,600 * min(input_tokens, 1,023)
```

An 8,192-token request at page size 1 is 719.609375 MiB aggregate. TP2
prefill to TP1 decode packs 359.8046875 MiB from each writer into one
719.609375 MiB decode lease. At page sizes 2, 16, and 64, the 1,023 logical
SWA rows occupy 1,024 physical rows, so the physical lease is 720 MiB while
the logical payload remains 719.609375 MiB.

## Required invariants

1. A component is addressed by its own exact source and destination page
   arrays. Main-KV indices are never reused for SWA, and destination pages are
   never reconstructed from `chunk_id`, `page_start`, or sequence offsets.
2. One page means `page_size` complete token rows. A partial final logical page
   transfers its complete physical page on both sides.
3. Every component span carries both logical and physical token counts.
   `logical_token_count <= physical_token_count`,
   `physical_token_count % page_size == 0`, and their difference is padding,
   never additional attention history.
4. Every per-entry item length is positive and divisible by `page_size`.
5. For every paired entry:

   ```text
   source_token_bytes * source_tp == destination_token_bytes * destination_tp
   ```

   `compute_tensor_parallel_shard` defines the only permitted byte routing.
   Replicated or unrelated rank pairs fail.
6. Bootstrap routing presents a prefill rank only to connected decode ranks.
   `_send_tp_sharded_state` must continue to reject unrelated pairs instead of
   returning an empty mapping. An empty result would conceal a routing bug.
7. A unique writer notification is counted once. Duplicate notifications never
   advance scatter readiness.
8. Source scratch cannot be rewritten until every NIXL handle reading that
   span is `DONE`.
9. A decode lease cannot be reused until the final scatter CUDA event for all
   of its components has completed.
10. Request KV pages remain allocated until every staged scatter has completed.
11. Allocation, layout, and notification metadata are immutable after the
    plan handshake.
12. No writer may issue a NIXL write before decode-side consensus commits the
    plan to every expected writer. After commit, an aborted lease is not reusable
    until every writer is quiesced or its transport epoch is revoked.

## Data model

Add the pure layout model in
`python/sglang/srt/disaggregation/common/staging_layout.py`.

```python
@dataclasses.dataclass(frozen=True)
class StagingComponentId:
    state_index: int | None
    state_type: StateType | None


@dataclasses.dataclass(frozen=True)
class StagingComponentSpan:
    component_id: StagingComponentId
    source_index_offset: int
    destination_index_offset: int
    logical_token_count: int
    physical_token_count: int


@dataclasses.dataclass(frozen=True)
class StagingWriterId:
    transfer_source_rank: int
    source_attn_tp_rank: int
    source_pp_rank: int
    source_cp_rank: int


@dataclasses.dataclass(frozen=True)
class StagingCopyGroup:
    component_id: StagingComponentId
    source_entry_indices: tuple[int, ...]
    destination_entry_indices: tuple[int, ...]
    packed_offset: int
    page_count: int
    source_token_bytes: int
    destination_token_bytes: int
    source_offset_bytes: int
    destination_offset_bytes: int
    copy_bytes_per_token: int


@dataclasses.dataclass(frozen=True)
class StagingWriterLayout:
    writer_id: StagingWriterId
    lease_offset: int
    length_bytes: int
    copy_groups: tuple[StagingCopyGroup, ...]


@dataclasses.dataclass(frozen=True)
class StagingChunkLayout:
    chunk_id: int
    is_last: bool
    component_spans: tuple[StagingComponentSpan, ...]
    writers: tuple[StagingWriterLayout, ...]
    total_bytes: int
    digest: bytes
```

`state_index=None` identifies main KV. A non-negative `state_index` identifies
the exact position in `KVArgs.state_types`, which permits more than one SWA
component without conflating them.

Add the runtime buffer registry in
`python/sglang/srt/disaggregation/common/staging_buffer.py`.

```python
@dataclasses.dataclass(frozen=True)
class StagingComponentBuffers:
    component_id: StagingComponentId
    tensors: tuple[torch.Tensor, ...]
    item_lens: tuple[int, ...]
    layer_ids: tuple[int, ...]
    page_size: int
```

Tensor order is the registered transfer-entry order, normally all K layers
followed by all V layers. Registration validates tensor pointers, row strides,
item lengths, layer ids, NHD layout, and the `KVArgs` pointer order once.
Pointer tensors used by Triton are created once and retained by the registry.

The layout builder is CPU-only and proportional to transfer entries, not
tokens:

```python
def build_staging_chunk_layout(
    *,
    chunk_id: int,
    is_last: bool,
    spans: tuple[StagingComponentSpan, ...],
    source_components: tuple[StagingComponentGeometry, ...],
    destination_components: tuple[StagingComponentGeometry, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    writers: tuple[StagingWriterId, ...],
) -> StagingChunkLayout:
    ...
```

It pairs entries by `(component_id, tensor occurrence, global layer id)`,
calls `compute_tensor_parallel_shard` for each connected writer, groups equal
copy geometry, aligns regions, and returns a deterministic digest. The digest
includes both token counts. Copy groups derive
`page_count = physical_token_count // page_size`; kernels copy the physical
count while scheduling and correctness use the logical count.

## Packed layout

The decode allocation is writer-major, then component-major:

```text
decode lease
  writer TP0
    main KV: K layers, V layers, all main pages
    SWA:     K layers, V layers, all final-window pages
  writer TP1
    main KV: K layers, V layers, all main pages
    SWA:     K layers, V layers, all final-window pages
```

Each prefill writer gathers only its `StagingWriterLayout` into a contiguous
source scratch span, then issues one NIXL VRAM write to its corresponding
contiguous decode span. The final chunk therefore uses one source descriptor
and one destination descriptor per writer for both full KV and SWA.

For Gemma 4 TP2 to TP1 at 8,192 tokens:

```text
page size 1
per writer main         = 160 MiB
per writer logical SWA  = 199.8046875 MiB
per writer span         = 359.8046875 MiB
decode lease            = 719.609375 MiB
```

At page size 2 or greater in the initial test set, the physical SWA capacity
rounds to 200 MiB per TP2 writer. The copy lease is then 360 MiB per writer and
720 MiB aggregate, but its manifest still records 1,023 logical SWA rows.

The gather and scatter implementation groups tensors with identical row
geometry. Gemma 4 needs one main group and one SWA group. Two small Triton
launches are preferable to a ragged mega-kernel until profiling proves launch
overhead material. Descriptor construction remains O(1).

## Exact page arrays

The runtime payload passed to gather is:

```python
@dataclasses.dataclass(frozen=True)
class StagingSourcePayload:
    layout: StagingChunkLayout
    source_pages: dict[StagingComponentId, np.ndarray]


@dataclasses.dataclass(frozen=True)
class StagingDestinationPayload:
    layout: StagingChunkLayout
    destination_pages: dict[StagingComponentId, np.ndarray]
```

The decode receiver retains the exact `kv_indices` and each exact
`state_indices[i]` array supplied to `send_metadata`. Each lease stores slices
of those arrays. `_scatter_region` must not index
`req_to_token_pool.req_to_token` using a delta-relative `page_start`.

The prefill worker uses `TransferKVChunk.prefill_kv_indices` for main KV and
`TransferKVChunk.state_indices[i]` for SWA. Source and destination page counts
must equal `physical_token_count / page_size` from the immutable component
span. The final page's padding is copied because it shares the physical page;
it is excluded from logical sequence lengths and attention.

A decode radix-cache full hit can produce zero main pages and non-empty SWA
pages. Such a request has one state-only final chunk whose logical count is at
most 1,023 and whose physical count is page-rounded:

```text
component_spans = [
  SWA(
    state_index=i,
    logical_token_count>0,
    physical_token_count>=logical_token_count,
  )
]
```

It receives a normal lease, notification, scatter event, and completion gate.

## Writer-scoped plan consensus

Chunk arithmetic has one decode-side authority. Every source writer constructs
the same immutable `StagingRequestPlan` after bootstrap, then submits only its
writer-scoped projection. There is no leader writer and no writer-to-writer
response forwarding.

For each destination room:

1. Decode retains the exact main and state destination arrays sent in normal
   metadata and derives `expected_writers` from the bootstrap TP mapping.
2. Every expected source writer sends exactly one `STAGING_PLAN_REQ` containing
   `(room, generation, writer_id)`, source and destination TP geometry, the
   global chunk-set digest, and that writer's manifest for every chunk. TP4 to
   TP1 therefore produces four independent submissions to the TP1 decode rank.
3. Decode stores submissions in `PendingStagingPlan`, keyed by
   `(room, generation)`. An exact duplicate from the same writer is idempotent.
   A divergent duplicate, unknown writer, stale generation, or unrelated rank
   fails the plan.
4. Decode does not allocate a lease or return a usable address until submissions
   from the complete expected-writer set have arrived. It canonicalizes writers
   by `StagingWriterId` and verifies consensus:
   - room, generation, chunk ids, final-chunk markers, TP geometry, component
     identities, and logical and physical token counts are identical;
   - the global digest recomputed from the canonical union matches every
     submitted digest;
   - each writer's source and destination byte ranges equal
     `compute_tensor_parallel_shard` for that writer;
   - writer ranges are disjoint and their union exactly covers each destination
     component span;
   - every range validates against decode's retained destination arrays.
5. After consensus, decode allocates each `DecodeStagingLease` once and creates
   one writer-specific `STAGING_PLAN_RESP(PREPARED)` for every expected writer.
   A response contains the common generation and digest, a plan capability, and
   only that writer's `(chunk_id, offset, size, round, end)` records. Decode
   caches responses so a retried request receives the identical response.
6. Each writer validates its own prepared response and sends
   `STAGING_PLAN_READY`. It may gather source data, but it must not issue NIXL.
7. When the ready-writer set equals the expected-writer set, decode sends
   `STAGING_PLAN_COMMIT` to every writer. Only this message authorizes NIXL
   writes. Notifications include the generation, digest, and capability.

The prepare and commit control traffic is once per request, not once per chunk,
and can overlap the prefill forward pass.

### Consensus failure and ownership

Before commit, any disagreement, timeout, response failure, or negative writer
readiness fails the room atomically. Decode sends a terminal
`STAGING_PLAN_RESP(ABORTED)` to every expected writer endpoint and releases its
leases immediately because no writer was authorized to write.

After commit, decode owns and quarantines every lease until success or explicit
writer quiescence:

1. Decode sends `STAGING_PLAN_ABORT` to all expected writers.
2. Each writer stops posting new writes, waits for every already-posted NIXL
   handle for that plan to become `DONE` or be cancelled, releases its source
   scratch, and sends `STAGING_WRITER_QUIESCED`.
3. Decode releases or reuses the remote lease only when the quiesced-writer set
   equals the expected-writer set.
4. If a writer or connection dies before quiescing, the lease remains
   quarantined until that writer's NIXL agent/registration epoch is revoked.
   A timeout alone never makes one-sided target memory safe to reuse.

All control records are idempotent under
`(room, generation, writer_id, digest)`. Terminal state is monotonic. This
protocol gives every writer a response, prevents early partial writes, and
makes ownership explicit for TP4 to TP1 rather than hoping one rank acts as an
undeclared coordinator.

## Writer identity and completion

NIXL staging notifications carry a fixed binary record, not an
underscore-delimited agent name:

```python
@dataclasses.dataclass(frozen=True)
class StagingWriteNotification:
    room: int
    generation: int
    chunk_id: int
    writer_id: StagingWriterId
    layout_digest: bytes
    plan_capability: bytes
```

`source_attn_tp_rank` selects byte placement. The complete
`StagingWriterId` deduplicates arrivals across TP, PP, and CP. The decode lease
holds `expected_writers` and `arrived_writers` sets. Scatter is submitted only
when the sets are equal and the notification generation, digest, and capability
match the committed plan.

The bootstrap mapping already prefilters peers:

* TP4 prefill to TP2 decode maps decode rank 0 to prefill ranks 0 and 1, and
  decode rank 1 to prefill ranks 2 and 3.
* TP2 prefill to TP4 decode maps decode ranks 0 and 1 to prefill rank 0, and
  decode ranks 2 and 3 to prefill rank 1.

The transport method treats an unrelated pair as a contract violation.

A staged final notification marks every included state component complete for
that writer. `maybe_send_extra` receives the set of staged state indices and
sends only remaining components. `TransferStatus` tracks state completion by
`(writer_id, state_index)`, rather than treating the first state notification
as completion for all state components.

## Buffer ownership and lifetime

### Prefill scratch

Each transfer worker owns a `SourceStagingArena`. A gather acquires a
`SourceStagingLease`; every destination peer written from that span is attached
to the lease. The lease is released only after all NIXL handles are `DONE`.

When one source rank feeds multiple decode ranks, their packed spans are
disjoint. The sum of those slices is at most the source rank's local component
bytes for a non-replicated TP partition. The worker can post all connected
peer writes concurrently without rewriting in-flight memory.

The gather stream waits on an explicit CUDA readiness event recorded after KV
production. It does not assume the producer used the default stream. NIXL is
posted only after the gather event completes.

### Decode ring

`DecodeStagingLease` owns:

```text
allocator id, byte range, round, component layout, exact destination pages,
generation, plan capability, expected writers, ready writers, arrived writers,
quiesced writers, scatter event, release state
```

NIXL completion makes the writer span visible. Once all unique writers arrive,
the scatter stream launches one kernel per copy group and records one final
event. The allocator frees the lease and advances the watermark only after that
event completes.

Abort and failure synchronize the scatter stream and follow the pre-commit or
post-commit quiescence rule before releasing each lease exactly once. Request KV
pages are freed only after the same ownership barrier. A successful request is
not visible to decode scheduling until transport status, all state components,
and all scatter events are complete.

## File-level implementation map

* `common/staging_layout.py`
  * Add the immutable component, span, writer, copy-group, chunk-layout, and
    digest types.
  * Add `build_staging_chunk_layout`.
* `common/staging_buffer.py`
  * Retain `StagingBuffer` and the ring allocator.
  * Replace head-specific layout math with byte-stride component gather and
    scatter groups.
  * Add `SourceStagingArena` and source lease tracking.
* `common/staging_handler.py`
  * Replace tuple-based `chunk_staging_infos` with typed decode leases.
  * Add writer-scoped plan collection, canonical consensus, prepared responses,
    ready/commit handling, writer-set arrival handling, quiescence, event
    polling, and exact-once release.
  * Remove per-chunk prefetch request state.
* `nixl/conn.py`
  * Register component buffers, send one contiguous write per writer layout,
    encode/decode writer notifications, and exclude staged state indices from
    `maybe_send_extra`.
  * Preserve the direct `_send_tp_sharded_state` path as the reference path.
* `prefill.py` and `decode.py`
  * Register full and SWA tensor components explicitly.
  * Retain exact component page arrays for payload construction.
* `mem_cache/swa_memory_pool.py`
  * Expose SWA K/V transfer tensors and global layer ids in registration order.

## Failure and fallback semantics

Staging selection is request-atomic.

* With staging disabled, main KV uses the direct slice path and SWA uses the
  direct TP-sharded reference path.
* With staging enabled, unsupported layout, replicated TP geometry, component
  mismatch, negative page index, insufficient per-chunk ring capacity, or plan
  digest mismatch fails before any staged write.
* No NIXL write is legal before `STAGING_PLAN_COMMIT`. A post-commit failure
  quarantines the lease until all writers quiesce or their transport epochs are
  revoked.
* No chunk silently falls back after allocation or after another component has
  been staged. Mixing paths would leak leases, corrupt completion accounting,
  or expose partially reconstructed KV.
* HND, page-major, mixed host/VRAM, PP>1, and CP>1 remain explicit
  initialization errors until their ownership and entry-routing contracts are
  implemented and tested.

## Required tests

### Pure layout and protocol tests

1. Gemma 4 TP2 to TP1 at page size 1 has two 359.8046875 MiB writer spans and
   a 719.609375 MiB lease at 8,192 tokens.
2. The same request at page sizes 2, 16, and 64 records 1,023 logical SWA rows,
   1,024 physical SWA rows, two 360 MiB spans, and a 720 MiB lease.
3. TP4 to TP2 and TP2 to TP4 produce exact connected writer sets, source
   offsets, and destination offsets.
4. Page sizes 1, 2, 16, and 64 use the exact physical token count,
   including a partial final logical page.
5. Main plus SWA final chunks, main-only intermediate chunks, and SWA-only
   final chunks produce stable digests and exact spans.
6. Duplicate, unknown, and out-of-order writer notifications never submit an
   early or repeated scatter.
7. TP4 to TP1 allocates nothing before four writer-scoped submissions reach
   consensus, emits four prepared responses and four commits, and accepts one
   writer notification per committed writer.
8. Manifest disagreement, missing writer, response loss, and ready timeout send
   terminal failure to every writer and expose no writable lease.
9. Post-commit abort does not reuse a lease before every writer quiesces; an
   unreachable writer quarantines the range until epoch revocation.

### GPU copy tests

1. Fill source pages with per-component, layer, page, token, and rank patterns.
2. Gather and scatter TP2 to TP1, TP4 to TP2, and TP2 to TP4 with non-contiguous
   page arrays.
3. Verify every destination byte and verify sentinels outside each destination
   shard remain unchanged.
4. Compare Triton and torch reference kernels for page sizes greater than one.

### Lifetime tests

1. A pending NIXL handle prevents source scratch reuse.
2. Two connected decode peers use disjoint source spans until both handles are
   done.
3. Decode ring wrap waits for the prior-round watermark.
4. Scatter event completion, not notification arrival, releases a decode
   lease.
5. Abort during gather, RDMA, and scatter releases every owned resource once
   after the required writer-quiescence barrier.

### Live integration

Run TP2 prefill to TP1 decode against deterministic TP1 aggregate output at
prompt lengths below 1,024, exactly 1,024, just above 1,024, 4,096, and near
8,192. Include page sizes 1 and 64. Record logical rows, physical rows, layout
time, plan-consensus time, prepare/commit time, gather time, NIXL time, scatter
time, descriptor counts, and bytes by component separately.
