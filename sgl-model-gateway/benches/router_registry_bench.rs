use std::{collections::HashMap, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, HttpOrigin, KvTransferProtocol, PdMetadataSchema,
    PdProcessMetadata, PdProcessRegistration, PdProcessRole, PreparedGrantProtocol, WorkerRegistry,
    WorkerType,
};
use uuid::Uuid;

// Helper to populate registry
fn setup_registry(count: usize) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());

    for i in 0..count {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "benchmark-model".to_string());

        let worker_type = if i % 2 == 0 {
            WorkerType::Regular
        } else {
            WorkerType::Decode
        };

        let url = format!("http://worker-{i}:8000");
        let mut builder = BasicWorkerBuilder::new(&url)
            .worker_type(worker_type)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default());
        if i % 2 != 0 {
            let metadata = PdProcessMetadata::new(
                PdMetadataSchema::V1,
                Uuid::from_u128(i as u128 + 1),
                PdProcessRole::Decode,
                1,
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "bf16",
                64,
                KvTransferProtocol::PackedV4,
                PreparedGrantProtocol::V1,
                None,
            )
            .expect("benchmark PD metadata must be valid");
            builder = builder.pd_process(PdProcessRegistration::new(
                HttpOrigin::parse(&url).expect("benchmark worker URL must be a valid HTTP origin"),
                metadata,
            ));
        }
        let worker = builder.build();

        registry
            .register(Arc::from(worker))
            .expect("benchmark worker registration must succeed");
    }
    registry
}

fn bench_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Registry Optimizations");

    // We test with 5000 workers to simulate high load
    let size = 5000;
    let registry = setup_registry(size);

    //  The OLD method (Slow: Allocates vector + Clones ARCs)
    group.bench_function(BenchmarkId::new("Old: get_all()", size), |b| {
        b.iter(|| {
            black_box(registry.get_all());
        });
    });

    //  The NEW method (Fast: O(1) Lookup, Zero Allocation)
    group.bench_function(
        BenchmarkId::new("New: get_worker_distribution()", size),
        |b| {
            b.iter(|| {
                black_box(registry.get_worker_distribution());
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_optimizations);
criterion_main!(benches);
