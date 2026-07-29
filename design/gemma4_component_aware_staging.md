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
| SWA | 50 | 16 heads x 256 dimensions | Final page-aligned 1024-token window |

With one-byte FP8 KV storage and page size 1, aggregate bytes are:

```text
40,960 * input_tokens + 409,600 * min(input_tokens, 1,024)
```

An 8,192-token request is 720 MiB aggregate. TP2 prefill to TP1 decode packs
360 MiB from each writer into one 720 MiB decode lease.

## Required invariants

1. A component is addressed by its own exact source and destination page
   arrays. Main-KV indices are never reused for SWA, and destination pages are
   never reconstructed from `chunk_id`, `page_start`, or sequence offsets.
2. One page means `page_size` complete token rows. A partial final logical page
   transfers its complete physical page on both sides.
3. Every per-entry item length is positive and divisible by `page_size`.
4. For every paired entry:

   ```text
   source_token_bytes * source_tp == destination_token_bytes * destination_tp
   ```

   `compute_tensor_parallel_shard` defines the only permitted byte routing.
   Replicated or unrelated rank pairs fail.
5. Bootstrap routing presents a prefill rank only to connected decode ranks.
   `_send_tp_sharded_state` must continue to reject unrelated pairs instead of
   returning an empty mapping. An empty result would conceal a routing bug.
6. A unique writer notification is counted once. Duplicate notifications never
   advance scatter readiness.
7. Source scratch cannot be rewritten until every NIXL handle reading that
   span is `DONE`.
8. A decode lease cannot be reused until the final scatter CUDA event for all
   of its components has completed.
9. Request KV pages remain allocated until every staged scatter has completed.
10. Allocation, layout, and notification metadata are immutable after the
    plan handshake.

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
    page_count: int


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
copy geometry, aligns regions, and returns a deterministic digest.

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
per writer main = 160 MiB
per writer SWA  = 200 MiB
per writer span = 360 MiB
decode lease    = 720 MiB
```

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
must equal the immutable component span.

A decode radix-cache full hit can produce zero main pages and non-empty SWA
pages. Such a request has one state-only final chunk:

```text
component_spans = [SWA(state_index=i, page_count>0)]
```

It receives a normal lease, notification, scatter event, and completion gate.

## Plan handshake

Chunk arithmetic must have one owner. Prefill constructs an immutable
`StagingRequestPlan` after bootstrap, using the same chunk spans later consumed
to create `TransferKVChunk` objects.

1. Decode sends exact main and state destination indices in the normal metadata
   message.
2. Prefill builds every chunk manifest, including a state-only final chunk
   when needed, and sends one `STAGING_PLAN_REQ` for the room. The request
   contains writer identity, TP geometry, component spans, and a plan digest.
3. Decode validates the manifests against its retained exact destination
   arrays, allocates one `DecodeStagingLease` per chunk, and replies once with
   all `(chunk_id, offset, size, round, end, digest)` records.
4. Prefill validates the response digest and can transfer each chunk as soon as
   its source data is ready and the remote watermark permits the write.

This replaces per-chunk `STAGING_REQ` fan-out and mutable parallel arrays in
`StagingTransferInfo`. The single control round trip can overlap the prefill
forward pass.

## Writer identity and completion

NIXL staging notifications carry a fixed binary record, not an
underscore-delimited agent name:

```python
@dataclasses.dataclass(frozen=True)
class StagingWriteNotification:
    room: int
    chunk_id: int
    writer_id: StagingWriterId
    layout_digest: bytes
```

`source_attn_tp_rank` selects byte placement. The complete
`StagingWriterId` deduplicates arrivals across TP, PP, and CP. The decode lease
holds `expected_writers` and `arrived_writers` sets. Scatter is submitted only
when the sets are equal.

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
expected writers, arrived writers, scatter event, release state
```

NIXL completion makes the writer span visible. Once all unique writers arrive,
the scatter stream launches one kernel per copy group and records one final
event. The allocator frees the lease and advances the watermark only after that
event completes.

Abort and failure synchronize the scatter stream, release each lease exactly
once, and then allow request KV pages to be freed. A successful request is not
visible to decode scheduling until transport status, all state components, and
all scatter events are complete.

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
  * Add plan request/response handling, writer-set arrival handling, event
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
* No chunk silently falls back after allocation or after another component has
  been staged. Mixing paths would leak leases, corrupt completion accounting,
  or expose partially reconstructed KV.
* HND, page-major, mixed host/VRAM, PP>1, and CP>1 remain explicit
  initialization errors until their ownership and entry-routing contracts are
  implemented and tested.

## Required tests

### Pure layout and protocol tests

1. Gemma 4 TP2 to TP1 layout has two 360 MiB writer spans and a 720 MiB lease
   at 8,192 tokens.
2. TP4 to TP2 and TP2 to TP4 produce exact connected writer sets, source
   offsets, and destination offsets.
3. Page sizes 1, 2, 16, and 64 use `page_count * page_size` token rows,
   including a partial final logical page.
4. Main plus SWA final chunks, main-only intermediate chunks, and SWA-only
   final chunks produce stable digests and exact spans.
5. Duplicate, unknown, and out-of-order writer notifications never submit an
   early or repeated scatter.
6. Manifest disagreement between writers fails the room before allocation is
   exposed.

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
5. Abort during gather, RDMA, and scatter releases every owned resource once.

### Live integration

Run TP2 prefill to TP1 decode against deterministic TP1 aggregate output at
prompt lengths below 1,024, exactly 1,024, just above 1,024, 4,096, and near
8,192. Record layout time, gather time, NIXL time, scatter time, descriptor
counts, and bytes by component separately.
