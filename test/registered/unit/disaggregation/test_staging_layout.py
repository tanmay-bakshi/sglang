import dataclasses
import unittest

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.staging_layout import (
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
    build_staging_chunk_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MEBIBYTE: int = 1024 * 1024
MAIN_KV: StagingComponentId = StagingComponentId(None, None)
SWA: StagingComponentId = StagingComponentId(0, StateType.SWA)
GEMMA4_FULL_LAYER_IDS: tuple[int, ...] = tuple(range(5, 60, 6))
GEMMA4_SWA_LAYER_IDS: tuple[int, ...] = tuple(
    layer_id for layer_id in range(60) if layer_id not in GEMMA4_FULL_LAYER_IDS
)


def make_writer(source_tp_rank: int) -> StagingWriterId:
    """Build one initial-contract writer identity.

    :param source_tp_rank: Source attention tensor-parallel rank.
    :returns: Writer identity.
    """

    return StagingWriterId(
        transfer_source_rank=source_tp_rank,
        source_attn_tp_rank=source_tp_rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )


def repeated_layer_ids(layer_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return K-then-V repeated layer identifiers.

    :param layer_ids: Global attention-layer identifiers.
    :returns: Layer identifiers for K and V registrations.
    """

    return layer_ids + layer_ids


def gemma4_geometry(
    *,
    component_id: StagingComponentId,
    layer_ids: tuple[int, ...],
    local_heads: int,
    head_dimension: int,
    page_size: int,
) -> StagingComponentGeometry:
    """Build a one-byte Gemma 4 KV component geometry.

    :param component_id: Component identity.
    :param layer_ids: Global attention-layer identifiers.
    :param local_heads: KV heads owned by the rank.
    :param head_dimension: Scalar dimension of each head.
    :param page_size: Complete token rows in one page.
    :returns: Registered component geometry.
    """

    item_len = local_heads * head_dimension * page_size
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=(item_len,) * (len(layer_ids) * 2),
        layer_ids=repeated_layer_ids(layer_ids),
        page_size=page_size,
    )


def gemma4_geometries(
    *,
    source_tp_size: int,
    destination_tp_size: int,
    page_size: int,
) -> tuple[
    tuple[StagingComponentGeometry, ...],
    tuple[StagingComponentGeometry, ...],
]:
    """Build source and destination Gemma 4 registration geometries.

    :param source_tp_size: Source attention TP width.
    :param destination_tp_size: Destination attention TP width.
    :param page_size: Complete token rows in one page.
    :returns: Source and destination component geometries.
    """

    source_components = (
        gemma4_geometry(
            component_id=MAIN_KV,
            layer_ids=GEMMA4_FULL_LAYER_IDS,
            local_heads=4 // source_tp_size,
            head_dimension=512,
            page_size=page_size,
        ),
        gemma4_geometry(
            component_id=SWA,
            layer_ids=GEMMA4_SWA_LAYER_IDS,
            local_heads=16 // source_tp_size,
            head_dimension=256,
            page_size=page_size,
        ),
    )
    destination_components = (
        gemma4_geometry(
            component_id=MAIN_KV,
            layer_ids=GEMMA4_FULL_LAYER_IDS,
            local_heads=4 // destination_tp_size,
            head_dimension=512,
            page_size=page_size,
        ),
        gemma4_geometry(
            component_id=SWA,
            layer_ids=GEMMA4_SWA_LAYER_IDS,
            local_heads=16 // destination_tp_size,
            head_dimension=256,
            page_size=page_size,
        ),
    )
    return source_components, destination_components


def build_gemma4_layout(
    *,
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    prefix_tokens: int,
    page_size: int,
    writers: tuple[StagingWriterId, ...],
    alignment_bytes: int = 256,
) -> StagingChunkLayout:
    """Build a full-attention plus SWA Gemma 4 layout.

    :param source_tp_size: Source attention TP width.
    :param destination_tp_size: Destination attention TP width.
    :param destination_tp_rank: Destination attention TP rank.
    :param prefix_tokens: Full-attention prefix length.
    :param page_size: Complete token rows in one page.
    :param writers: Connected writers.
    :param alignment_bytes: Packed-region byte alignment.
    :returns: Packed chunk layout.
    """

    source_components, destination_components = gemma4_geometries(
        source_tp_size=source_tp_size,
        destination_tp_size=destination_tp_size,
        page_size=page_size,
    )
    swa_logical_tokens = min(prefix_tokens, 1023)
    swa_physical_tokens = (
        (swa_logical_tokens + page_size - 1) // page_size
    ) * page_size
    full_physical_tokens = ((prefix_tokens + page_size - 1) // page_size) * page_size
    return build_staging_chunk_layout(
        chunk_id=7,
        is_last=True,
        spans=(
            StagingComponentSpan(
                component_id=MAIN_KV,
                source_index_offset=0,
                destination_index_offset=0,
                logical_token_count=prefix_tokens,
                physical_token_count=full_physical_tokens,
            ),
            StagingComponentSpan(
                component_id=SWA,
                source_index_offset=0,
                destination_index_offset=0,
                logical_token_count=swa_logical_tokens,
                physical_token_count=swa_physical_tokens,
            ),
        ),
        source_components=source_components,
        destination_components=destination_components,
        source_tp_size=source_tp_size,
        destination_tp_size=destination_tp_size,
        destination_tp_rank=destination_tp_rank,
        writers=writers,
        alignment_bytes=alignment_bytes,
    )


class TestGemma4StagingLayout(unittest.TestCase):
    """Tests exact packed sizes and TP byte placement for Gemma 4."""

    def test_fixture_uses_global_layers_and_component_local_offsets(self) -> None:
        """The fixture models Gemma's real layer pattern and separate page arrays."""

        source_components, _ = gemma4_geometries(
            source_tp_size=2,
            destination_tp_size=1,
            page_size=64,
        )
        layout = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=8192,
            page_size=64,
            writers=(make_writer(0), make_writer(1)),
        )

        self.assertEqual(
            source_components[0].layer_ids,
            repeated_layer_ids(GEMMA4_FULL_LAYER_IDS),
        )
        self.assertEqual(
            source_components[1].layer_ids,
            repeated_layer_ids(GEMMA4_SWA_LAYER_IDS),
        )
        self.assertEqual(
            [
                (span.source_index_offset, span.destination_index_offset)
                for span in layout.component_spans
            ],
            [(0, 0), (0, 0)],
        )

    def test_tp2_to_tp1_page_one_uses_two_bulk_writer_spans(self) -> None:
        """Each writer owns 359.8046875 MiB of the 719.609375 MiB lease."""

        layout = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=8192,
            page_size=1,
            writers=(make_writer(1), make_writer(0)),
        )

        self.assertEqual(layout.total_bytes, 719 * MEBIBYTE + 638_976)
        self.assertEqual(len(layout.writers), 2)
        self.assertEqual(
            [writer.writer_id.source_attn_tp_rank for writer in layout.writers],
            [0, 1],
        )
        self.assertEqual(
            [writer.length_bytes for writer in layout.writers],
            [359 * MEBIBYTE + 843_776, 359 * MEBIBYTE + 843_776],
        )
        self.assertEqual(
            [len(writer.copy_groups) for writer in layout.writers],
            [2, 2],
        )
        first_writer_main, first_writer_swa = layout.writers[0].copy_groups
        second_writer_main, second_writer_swa = layout.writers[1].copy_groups
        self.assertEqual(first_writer_main.destination_offset_bytes, 0)
        self.assertEqual(first_writer_swa.destination_offset_bytes, 0)
        self.assertEqual(second_writer_main.destination_offset_bytes, 1024)
        self.assertEqual(second_writer_swa.destination_offset_bytes, 2048)
        self.assertEqual(first_writer_main.length_bytes, 160 * MEBIBYTE)
        self.assertEqual(first_writer_swa.length_bytes, 199 * MEBIBYTE + 843_776)

    def test_page_rounding_preserves_1023_logical_swa_rows(self) -> None:
        """Complete pages copy 1,024 SWA rows without expanding attention."""

        for page_size in (2, 16, 64):
            with self.subTest(page_size=page_size):
                layout = build_gemma4_layout(
                    source_tp_size=2,
                    destination_tp_size=1,
                    destination_tp_rank=0,
                    prefix_tokens=8192,
                    page_size=page_size,
                    writers=(make_writer(0), make_writer(1)),
                )

                self.assertEqual(layout.total_bytes, 720 * MEBIBYTE)
                swa_span = next(
                    span for span in layout.component_spans if span.component_id == SWA
                )
                self.assertEqual(swa_span.logical_token_count, 1023)
                self.assertEqual(swa_span.physical_token_count, 1024)
                self.assertEqual(
                    layout.writers[0].copy_groups[1].page_count,
                    1024 // page_size,
                )

    def test_tp4_writers_exactly_partition_destination_heads(self) -> None:
        """Four writers cover every destination token byte once."""

        layout = build_gemma4_layout(
            source_tp_size=4,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=1024,
            page_size=1,
            writers=tuple(make_writer(rank) for rank in range(4)),
        )

        main_offsets = [
            writer.copy_groups[0].destination_offset_bytes for writer in layout.writers
        ]
        swa_offsets = [
            writer.copy_groups[1].destination_offset_bytes for writer in layout.writers
        ]
        self.assertEqual(main_offsets, [0, 512, 1024, 1536])
        self.assertEqual(swa_offsets, [0, 1024, 2048, 3072])

    def test_tp8_replication_is_derived_per_component_entry(self) -> None:
        """Full KV uses even owners while every rank still carries SWA."""

        source_components = (
            StagingComponentGeometry(
                component_id=MAIN_KV,
                item_lens=(512, 512),
                layer_ids=(5, 5),
                page_size=1,
            ),
            StagingComponentGeometry(
                component_id=SWA,
                item_lens=(512, 512),
                layer_ids=(1, 1),
                page_size=1,
            ),
        )
        destination_components = (
            dataclasses.replace(
                source_components[0],
                item_lens=(2048, 2048),
            ),
            dataclasses.replace(
                source_components[1],
                item_lens=(4096, 4096),
            ),
        )
        spans = (
            StagingComponentSpan(MAIN_KV, 0, 0, 1, 1),
            StagingComponentSpan(SWA, 0, 0, 1, 1),
        )

        layout = build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=spans,
            source_components=source_components,
            destination_components=destination_components,
            source_tp_size=8,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=tuple(make_writer(rank) for rank in range(8)),
        )

        for rank, writer in enumerate(layout.writers):
            component_ids = tuple(
                group.component_id for group in writer.copy_groups
            )
            if rank % 2 == 0:
                self.assertEqual(component_ids, (MAIN_KV, SWA))
                self.assertEqual(
                    writer.copy_groups[0].destination_offset_bytes,
                    (rank // 2) * 512,
                )
            else:
                self.assertEqual(component_ids, (SWA,))
            self.assertEqual(
                writer.copy_groups[-1].destination_offset_bytes,
                rank * 512,
            )

    def test_tp8_replica_only_writer_gets_synchronization_projection(self) -> None:
        """A writer without payload retains one aligned transport projection."""

        source = StagingComponentGeometry(
            component_id=MAIN_KV,
            item_lens=(512, 512),
            layer_ids=(5, 5),
            page_size=1,
        )
        destination = dataclasses.replace(source, item_lens=(2048, 2048))
        layout = build_staging_chunk_layout(
            chunk_id=0,
            is_last=False,
            spans=(StagingComponentSpan(MAIN_KV, 0, 0, 1, 1),),
            source_components=(source,),
            destination_components=(destination,),
            source_tp_size=8,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=tuple(make_writer(rank) for rank in range(8)),
        )

        for rank, writer in enumerate(layout.writers):
            if rank % 2 == 0:
                self.assertEqual(len(writer.copy_groups), 1)
                self.assertEqual(writer.length_bytes, 1024)
                continue
            self.assertEqual(writer.copy_groups, ())
            self.assertEqual(writer.length_bytes, 256)

    def test_tp8_replication_can_differ_within_one_component(self) -> None:
        """Each paired registration entry derives its own physical owners."""

        source = StagingComponentGeometry(
            component_id=MAIN_KV,
            item_lens=(512, 512, 512, 512),
            layer_ids=(5, 7, 5, 7),
            page_size=1,
        )
        destination = dataclasses.replace(
            source,
            item_lens=(2048, 4096, 2048, 4096),
        )
        layout = build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=(StagingComponentSpan(MAIN_KV, 0, 0, 1, 1),),
            source_components=(source,),
            destination_components=(destination,),
            source_tp_size=8,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=tuple(make_writer(rank) for rank in range(8)),
        )

        for rank, writer in enumerate(layout.writers):
            source_entries = tuple(
                group.source_entry_indices for group in writer.copy_groups
            )
            if rank % 2 == 0:
                self.assertEqual(source_entries, ((0, 2), (1, 3)))
                continue
            self.assertEqual(source_entries, ((1, 3),))

    def test_tp1_to_tp2_selects_the_destination_half_of_each_source_token(
        self,
    ) -> None:
        """A destination rank gathers its half from the single source writer."""

        layout = build_gemma4_layout(
            source_tp_size=1,
            destination_tp_size=2,
            destination_tp_rank=1,
            prefix_tokens=256,
            page_size=1,
            writers=(make_writer(0),),
        )

        main_group, swa_group = layout.writers[0].copy_groups
        self.assertEqual(main_group.source_offset_bytes, 1024)
        self.assertEqual(main_group.destination_offset_bytes, 0)
        self.assertEqual(swa_group.source_offset_bytes, 2048)
        self.assertEqual(swa_group.destination_offset_bytes, 0)

    def test_tp4_to_tp2_selects_two_connected_writers(self) -> None:
        """One TP2 destination consumes exactly its two connected TP4 writers."""

        layout = build_gemma4_layout(
            source_tp_size=4,
            destination_tp_size=2,
            destination_tp_rank=1,
            prefix_tokens=1024,
            page_size=1,
            writers=(make_writer(3), make_writer(2)),
        )

        self.assertEqual(
            [writer.writer_id.source_attn_tp_rank for writer in layout.writers],
            [2, 3],
        )
        self.assertEqual(
            [
                writer.copy_groups[0].destination_offset_bytes
                for writer in layout.writers
            ],
            [0, 512],
        )
        self.assertEqual(
            [
                writer.copy_groups[1].destination_offset_bytes
                for writer in layout.writers
            ],
            [0, 1024],
        )

    def test_tp2_to_tp4_selects_one_destination_slice(self) -> None:
        """One TP4 destination consumes its half of the connected TP2 writer."""

        layout = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=4,
            destination_tp_rank=3,
            prefix_tokens=1024,
            page_size=1,
            writers=(make_writer(1),),
        )

        main_group, swa_group = layout.writers[0].copy_groups
        self.assertEqual(main_group.source_offset_bytes, 512)
        self.assertEqual(swa_group.source_offset_bytes, 1024)
        self.assertEqual(main_group.destination_offset_bytes, 0)
        self.assertEqual(swa_group.destination_offset_bytes, 0)

    def test_main_only_intermediate_chunk_omits_swa_storage(self) -> None:
        """Registered SWA geometry does not force it into intermediate chunks."""

        source_components, destination_components = gemma4_geometries(
            source_tp_size=2,
            destination_tp_size=1,
            page_size=1,
        )
        layout = build_staging_chunk_layout(
            chunk_id=2,
            is_last=False,
            spans=(
                StagingComponentSpan(
                    component_id=MAIN_KV,
                    source_index_offset=0,
                    destination_index_offset=0,
                    logical_token_count=2048,
                    physical_token_count=2048,
                ),
            ),
            source_components=source_components,
            destination_components=destination_components,
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=(make_writer(0), make_writer(1)),
        )

        self.assertEqual(layout.total_bytes, 80 * MEBIBYTE)
        self.assertEqual(
            tuple(span.component_id for span in layout.component_spans),
            (MAIN_KV,),
        )
        self.assertTrue(
            all(
                tuple(group.component_id for group in writer.copy_groups) == (MAIN_KV,)
                for writer in layout.writers
            )
        )

    def test_swa_only_final_chunk_omits_empty_main_storage(self) -> None:
        """A decode full hit can stage SWA without inventing main-KV pages."""

        source_components, destination_components = gemma4_geometries(
            source_tp_size=2,
            destination_tp_size=1,
            page_size=64,
        )
        layout = build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=(
                StagingComponentSpan(
                    component_id=SWA,
                    source_index_offset=0,
                    destination_index_offset=0,
                    logical_token_count=1023,
                    physical_token_count=1024,
                ),
            ),
            source_components=source_components,
            destination_components=destination_components,
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=(make_writer(0), make_writer(1)),
        )

        self.assertEqual(layout.total_bytes, 400 * MEBIBYTE)
        self.assertEqual(
            tuple(span.component_id for span in layout.component_spans),
            (SWA,),
        )
        self.assertTrue(
            all(
                tuple(group.component_id for group in writer.copy_groups) == (SWA,)
                for writer in layout.writers
            )
        )

    def test_digest_is_independent_of_writer_submission_order(self) -> None:
        """Canonical writer ordering produces one immutable plan digest."""

        first = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=4096,
            page_size=1,
            writers=(make_writer(0), make_writer(1)),
        )
        second = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=4096,
            page_size=1,
            writers=(make_writer(1), make_writer(0)),
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first.digest), 32)

    def test_digest_covers_alignment_even_when_offsets_are_unchanged(self) -> None:
        """Plan policy remains part of consensus when payload sizes align."""

        aligned_256 = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=4096,
            page_size=1,
            writers=(make_writer(0), make_writer(1)),
            alignment_bytes=256,
        )
        aligned_512 = build_gemma4_layout(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            prefix_tokens=4096,
            page_size=1,
            writers=(make_writer(0), make_writer(1)),
            alignment_bytes=512,
        )

        self.assertEqual(aligned_256.total_bytes, aligned_512.total_bytes)
        self.assertNotEqual(aligned_256.digest, aligned_512.digest)

    def test_digest_covers_topology_when_packed_bytes_are_unchanged(self) -> None:
        """Consensus distinguishes equal local layouts from different TP worlds."""

        geometry = gemma4_geometry(
            component_id=MAIN_KV,
            layer_ids=(0,),
            local_heads=2,
            head_dimension=8,
            page_size=2,
        )
        span = StagingComponentSpan(
            component_id=MAIN_KV,
            source_index_offset=0,
            destination_index_offset=0,
            logical_token_count=3,
            physical_token_count=4,
        )

        tp1 = build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=(span,),
            source_components=(geometry,),
            destination_components=(geometry,),
            source_tp_size=1,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=(make_writer(0),),
        )
        tp2 = build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=(span,),
            source_components=(geometry,),
            destination_components=(geometry,),
            source_tp_size=2,
            destination_tp_size=2,
            destination_tp_rank=0,
            writers=(make_writer(0),),
        )

        self.assertEqual(tp1.total_bytes, tp2.total_bytes)
        self.assertNotEqual(tp1.digest, tp2.digest)


class TestStagingLayoutValidation(unittest.TestCase):
    """Tests fail-closed component, page, and writer invariants."""

    def setUp(self) -> None:
        """Create a minimal TP2-to-TP1 component plan."""

        self.source = gemma4_geometry(
            component_id=MAIN_KV,
            layer_ids=(0,),
            local_heads=2,
            head_dimension=8,
            page_size=2,
        )
        self.destination = gemma4_geometry(
            component_id=MAIN_KV,
            layer_ids=(0,),
            local_heads=4,
            head_dimension=8,
            page_size=2,
        )
        self.span = StagingComponentSpan(
            component_id=MAIN_KV,
            source_index_offset=0,
            destination_index_offset=0,
            logical_token_count=3,
            physical_token_count=4,
        )

    def build(
        self,
        *,
        span: StagingComponentSpan | None = None,
        source: StagingComponentGeometry | None = None,
        destination: StagingComponentGeometry | None = None,
        writers: tuple[StagingWriterId, ...] | None = None,
    ) -> StagingChunkLayout:
        """Build the minimal validation fixture.

        :param span: Optional replacement span.
        :param source: Optional replacement source geometry.
        :param destination: Optional replacement destination geometry.
        :param writers: Optional replacement writer set.
        :returns: Packed chunk layout.
        """

        return build_staging_chunk_layout(
            chunk_id=0,
            is_last=True,
            spans=(span if span is not None else self.span,),
            source_components=(source if source is not None else self.source,),
            destination_components=(
                destination if destination is not None else self.destination,
            ),
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=(
                writers if writers is not None else (make_writer(0), make_writer(1))
            ),
        )

    def test_rejects_non_page_rounded_physical_count(self) -> None:
        """Physical capacity must be the exact page-rounded logical count."""

        invalid_span = dataclasses.replace(self.span, physical_token_count=6)
        with self.assertRaisesRegex(ValueError, "exact page-rounded"):
            self.build(span=invalid_span)

    def test_rejects_empty_component_span(self) -> None:
        """An included component must own at least one logical token."""

        invalid_span = dataclasses.replace(
            self.span,
            logical_token_count=0,
            physical_token_count=0,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.build(span=invalid_span)

    def test_rejects_chunk_without_components(self) -> None:
        """An empty chunk cannot produce a zero-byte transport lease."""

        with self.assertRaisesRegex(ValueError, "at least one component span"):
            build_staging_chunk_layout(
                chunk_id=0,
                is_last=True,
                spans=(),
                source_components=(self.source,),
                destination_components=(self.destination,),
                source_tp_size=2,
                destination_tp_size=1,
                destination_tp_rank=0,
                writers=(make_writer(0), make_writer(1)),
            )

    def test_rejects_missing_writer(self) -> None:
        """The writer set must cover every destination byte range."""

        with self.assertRaisesRegex(ValueError, "bootstrap routing"):
            self.build(writers=(make_writer(0),))

    def test_rejects_duplicate_source_attention_rank(self) -> None:
        """Two transport identities cannot claim one source TP shard."""

        duplicate = dataclasses.replace(make_writer(0), transfer_source_rank=9)
        with self.assertRaisesRegex(ValueError, "source attention TP ranks"):
            self.build(writers=(make_writer(0), duplicate))

    def test_rejects_duplicate_transfer_source_rank(self) -> None:
        """A global transport identity cannot own two source TP shards."""

        duplicate_transport_rank = dataclasses.replace(
            make_writer(1),
            transfer_source_rank=0,
        )
        with self.assertRaisesRegex(ValueError, "transfer source ranks"):
            self.build(writers=(make_writer(0), duplicate_transport_rank))

    def test_rejects_mismatched_repeated_layer_entries(self) -> None:
        """K/V occurrence pairing cannot silently use positional fallback."""

        destination = dataclasses.replace(
            self.destination,
            layer_ids=(0, 1),
        )
        with self.assertRaisesRegex(ValueError, "different layer entries"):
            self.build(destination=destination)

    def test_pairing_preserves_source_registration_order(self) -> None:
        """Packing follows source tensor order while matching destination IDs."""

        source = dataclasses.replace(
            self.source,
            item_lens=(32, 32, 32, 32),
            layer_ids=(0, 1, 0, 1),
        )
        destination = dataclasses.replace(
            self.destination,
            item_lens=(64, 64, 64, 64),
            layer_ids=(1, 0, 1, 0),
        )

        layout = self.build(source=source, destination=destination)
        group = layout.writers[0].copy_groups[0]

        self.assertEqual(group.source_entry_indices, (0, 1, 2, 3))
        self.assertEqual(group.destination_entry_indices, (1, 0, 3, 2))

    def test_rejects_negative_global_layer_identifier(self) -> None:
        """Registration identities use non-negative global model layers."""

        source = dataclasses.replace(self.source, layer_ids=(-1, -1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.build(source=source)

    def test_rejects_incompatible_aggregate_token_geometry(self) -> None:
        """Replicated or partial tensor geometry is not a TP partition."""

        destination = dataclasses.replace(
            self.destination,
            item_lens=(48, 48),
        )
        with self.assertRaisesRegex(ValueError, "exact source replication"):
            self.build(destination=destination)

    def test_rejects_state_type_without_state_index(self) -> None:
        """Main KV identity cannot carry an auxiliary state type."""

        invalid_component = StagingComponentId(None, StateType.SWA)
        source = dataclasses.replace(self.source, component_id=invalid_component)
        destination = dataclasses.replace(
            self.destination,
            component_id=invalid_component,
        )
        span = dataclasses.replace(self.span, component_id=invalid_component)
        with self.assertRaisesRegex(ValueError, "main KV"):
            self.build(span=span, source=source, destination=destination)

    def test_rejects_two_state_types_at_one_state_index(self) -> None:
        """One KVArgs state position cannot identify two components."""

        swa_component = StagingComponentId(0, StateType.SWA)
        dsa_component = StagingComponentId(0, StateType.DSA)
        source_swa = dataclasses.replace(self.source, component_id=swa_component)
        source_dsa = dataclasses.replace(self.source, component_id=dsa_component)
        destination_swa = dataclasses.replace(
            self.destination,
            component_id=swa_component,
        )
        destination_dsa = dataclasses.replace(
            self.destination,
            component_id=dsa_component,
        )
        span = dataclasses.replace(self.span, component_id=swa_component)

        with self.assertRaisesRegex(ValueError, "state_index 0"):
            build_staging_chunk_layout(
                chunk_id=0,
                is_last=True,
                spans=(span,),
                source_components=(source_swa, source_dsa),
                destination_components=(destination_swa, destination_dsa),
                source_tp_size=2,
                destination_tp_size=1,
                destination_tp_rank=0,
                writers=(make_writer(0), make_writer(1)),
            )

    def test_digest_covers_global_layer_identity(self) -> None:
        """Equal entry offsets do not erase the registered layer contract."""

        original = self.build()
        source = dataclasses.replace(self.source, layer_ids=(9, 9))
        destination = dataclasses.replace(self.destination, layer_ids=(9, 9))
        relabeled = self.build(source=source, destination=destination)

        self.assertEqual(original.total_bytes, relabeled.total_bytes)
        self.assertNotEqual(original.digest, relabeled.digest)


if __name__ == "__main__":
    unittest.main()
