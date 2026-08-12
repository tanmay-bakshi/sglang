//! Atomic reservation substrate for predicted prefill completion routing.
//!
//! This module deliberately owns neither a service-time curve nor a batching
//! model. The caller supplies both the per-request service cost and a predictor
//! derived from measured behavior. That keeps request ownership and concurrent
//! charging usable while the production prediction model remains undecided.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
use thiserror::Error;

use super::{pd_decoder_grant::PrefillId, PdGroupId};

/// Exact topology group and prefill process generation eligible for routing.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PdPrefillRoute {
    group_id: PdGroupId,
    prefill_id: PrefillId,
}

impl PdPrefillRoute {
    /// Construct an exact prefill routing identity.
    pub fn new(group_id: PdGroupId, prefill_id: PrefillId) -> Self {
        Self {
            group_id,
            prefill_id,
        }
    }

    /// Return the immutable topology group.
    pub fn group_id(&self) -> &PdGroupId {
        &self.group_id
    }

    /// Return the exact selected prefill process generation.
    pub fn prefill_id(&self) -> &PrefillId {
        &self.prefill_id
    }
}

/// One caller-qualified group considered by the completion predictor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdPrefillCandidate {
    route: PdPrefillRoute,
    manifest_index: usize,
    available: bool,
    service_cost: Duration,
}

impl PdPrefillCandidate {
    /// Construct one candidate from current process state and measured cost.
    pub fn new(
        route: PdPrefillRoute,
        manifest_index: usize,
        available: bool,
        service_cost: Duration,
    ) -> Self {
        Self {
            route,
            manifest_index,
            available,
            service_cost,
        }
    }

    /// Return the exact group and prefill generation.
    pub fn route(&self) -> &PdPrefillRoute {
        &self.route
    }

    /// Return the immutable topology order used for deterministic ties.
    pub fn manifest_index(&self) -> usize {
        self.manifest_index
    }

    /// Return whether the caller found the complete group eligible.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Return the caller-supplied service cost for this request.
    pub fn service_cost(&self) -> Duration {
        self.service_cost
    }
}

/// Read-only in-flight charge supplied to a completion predictor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdPrefillOutstandingCharge {
    reservation_id: u64,
    logical_request_id: Arc<str>,
    route: PdPrefillRoute,
    service_cost: Duration,
}

impl PdPrefillOutstandingCharge {
    /// Return the logical request that owns this charge.
    pub fn logical_request_id(&self) -> &str {
        &self.logical_request_id
    }

    /// Return the exact charged group and process generation.
    pub fn route(&self) -> &PdPrefillRoute {
        &self.route
    }

    /// Return the caller-supplied cost retained until terminal release.
    pub fn service_cost(&self) -> Duration {
        self.service_cost
    }
}

/// Measured policy for scoring one candidate under its current reservations.
pub trait PdPrefillCompletionModel: Send + Sync {
    /// Predict a comparable completion duration for one candidate.
    ///
    /// The scheduler invokes this method while holding its reservation lock.
    /// Implementations must be deterministic, non-blocking, and must not call
    /// back into the scheduler.
    fn predicted_completion(
        &self,
        candidate: &PdPrefillCandidate,
        outstanding: &[PdPrefillOutstandingCharge],
    ) -> Duration;
}

#[derive(Debug, Default)]
struct PdPrefillSchedulerState {
    next_reservation_id: u64,
    outstanding_by_group: HashMap<PdGroupId, Vec<PdPrefillOutstandingCharge>>,
    active_request_groups: HashMap<Arc<str>, PdGroupId>,
}

#[derive(Debug, Default)]
struct PdPrefillSchedulerInner {
    state: Mutex<PdPrefillSchedulerState>,
}

impl PdPrefillSchedulerInner {
    fn release(&self, reservation: &PdPrefillReservation) {
        let mut state = self.state.lock();
        let active_group = state
            .active_request_groups
            .remove(reservation.logical_request_id.as_ref());
        debug_assert_eq!(active_group.as_ref(), Some(reservation.route.group_id()));

        let remove_group = {
            let outstanding = state
                .outstanding_by_group
                .get_mut(reservation.route.group_id())
                .expect("a live prefill reservation retains its group ledger");
            let position = outstanding
                .iter()
                .position(|charge| charge.reservation_id == reservation.reservation_id)
                .expect("a live prefill reservation retains its exact charge");
            let charge = outstanding.remove(position);
            debug_assert_eq!(charge.logical_request_id, reservation.logical_request_id);
            debug_assert_eq!(charge.route, reservation.route);
            outstanding.is_empty()
        };
        if remove_group {
            state
                .outstanding_by_group
                .remove(reservation.route.group_id());
        }
    }
}

/// Shared authority for atomic predicted-completion selection and charging.
#[derive(Clone, Debug, Default)]
pub struct PdPrefillScheduler {
    inner: Arc<PdPrefillSchedulerInner>,
}

impl PdPrefillScheduler {
    /// Construct an empty scheduler authority.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the minimum predicted completion and atomically retain its charge.
    pub fn reserve(
        &self,
        logical_request_id: impl Into<String>,
        candidates: &[PdPrefillCandidate],
        model: &dyn PdPrefillCompletionModel,
    ) -> Result<PdPrefillReservation, PdPrefillSchedulerError> {
        validate_candidates(candidates)?;
        let logical_request_id: Arc<str> = Arc::from(logical_request_id.into());
        let mut state = self.inner.state.lock();
        if state
            .active_request_groups
            .contains_key(logical_request_id.as_ref())
        {
            return Err(PdPrefillSchedulerError::DuplicateLogicalRequest(
                logical_request_id.to_string(),
            ));
        }

        let mut selected: Option<(&PdPrefillCandidate, Duration)> = None;
        for candidate in candidates
            .iter()
            .filter(|candidate| candidate.is_available())
        {
            let outstanding = state
                .outstanding_by_group
                .get(candidate.route().group_id())
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let predicted_completion = model.predicted_completion(candidate, outstanding);
            let replace = selected
                .as_ref()
                .is_none_or(|(incumbent, incumbent_completion)| {
                    predicted_completion < *incumbent_completion
                        || (predicted_completion == *incumbent_completion
                            && candidate.manifest_index() < incumbent.manifest_index())
                });
            if replace {
                selected = Some((candidate, predicted_completion));
            }
        }
        let (candidate, predicted_completion) =
            selected.ok_or(PdPrefillSchedulerError::NoAvailableCandidate)?;

        let reservation_id = state
            .next_reservation_id
            .checked_add(1)
            .ok_or(PdPrefillSchedulerError::ReservationIdentityExhausted)?;
        state.next_reservation_id = reservation_id;
        let route = candidate.route().clone();
        state
            .outstanding_by_group
            .entry(route.group_id().clone())
            .or_default()
            .push(PdPrefillOutstandingCharge {
                reservation_id,
                logical_request_id: Arc::clone(&logical_request_id),
                route: route.clone(),
                service_cost: candidate.service_cost(),
            });
        state
            .active_request_groups
            .insert(Arc::clone(&logical_request_id), route.group_id().clone());

        Ok(PdPrefillReservation {
            inner: Arc::clone(&self.inner),
            reservation_id,
            logical_request_id,
            route,
            predicted_completion,
            active: true,
        })
    }

    #[cfg(test)]
    fn outstanding_reservations(&self, group_id: &PdGroupId) -> usize {
        self.inner
            .state
            .lock()
            .outstanding_by_group
            .get(group_id)
            .map_or(0, Vec::len)
    }
}

/// Exclusive terminal authority for one predicted prefill charge.
#[must_use = "a predicted prefill reservation must complete, cancel, or remain drop-owned"]
pub struct PdPrefillReservation {
    inner: Arc<PdPrefillSchedulerInner>,
    reservation_id: u64,
    logical_request_id: Arc<str>,
    route: PdPrefillRoute,
    predicted_completion: Duration,
    active: bool,
}

impl fmt::Debug for PdPrefillReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PdPrefillReservation")
            .field("logical_request_id", &self.logical_request_id)
            .field("route", &self.route)
            .field("predicted_completion", &self.predicted_completion)
            .field("active", &self.active)
            .finish()
    }
}

impl PdPrefillReservation {
    /// Return the logical request owning this reservation.
    pub fn logical_request_id(&self) -> &str {
        &self.logical_request_id
    }

    /// Return the exact selected group and prefill generation.
    pub fn route(&self) -> &PdPrefillRoute {
        &self.route
    }

    /// Return the score used for the atomic selection decision.
    pub fn predicted_completion(&self) -> Duration {
        self.predicted_completion
    }

    /// Release the charge after prefill service reaches its terminal point.
    pub fn complete(mut self) {
        self.release();
    }

    /// Release the charge after request cancellation.
    pub fn cancel(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.inner.release(self);
        self.active = false;
    }
}

impl Drop for PdPrefillReservation {
    fn drop(&mut self) {
        self.release();
    }
}

fn validate_candidates(candidates: &[PdPrefillCandidate]) -> Result<(), PdPrefillSchedulerError> {
    let mut group_ids = HashSet::new();
    let mut manifest_indices = HashSet::new();
    for candidate in candidates {
        if !group_ids.insert(candidate.route().group_id()) {
            return Err(PdPrefillSchedulerError::DuplicateGroupCandidate(
                candidate.route().group_id().clone(),
            ));
        }
        if !manifest_indices.insert(candidate.manifest_index()) {
            return Err(PdPrefillSchedulerError::DuplicateManifestIndex(
                candidate.manifest_index(),
            ));
        }
    }
    Ok(())
}

/// Invalid predicted-prefill reservation request.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PdPrefillSchedulerError {
    #[error("prefill candidate set contains topology group {0:?} more than once")]
    DuplicateGroupCandidate(PdGroupId),
    #[error("prefill candidate set contains manifest index {0} more than once")]
    DuplicateManifestIndex(usize),
    #[error("logical request {0} already owns a prefill routing reservation")]
    DuplicateLogicalRequest(String),
    #[error("no prefill candidate is currently available")]
    NoAvailableCandidate,
    #[error("prefill reservation identity space is exhausted")]
    ReservationIdentityExhausted,
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Barrier},
        thread,
    };

    use uuid::Uuid;

    use super::*;
    use crate::core::HttpOrigin;

    struct SerialCostModel;

    impl PdPrefillCompletionModel for SerialCostModel {
        fn predicted_completion(
            &self,
            candidate: &PdPrefillCandidate,
            outstanding: &[PdPrefillOutstandingCharge],
        ) -> Duration {
            outstanding
                .iter()
                .fold(candidate.service_cost(), |completion, charge| {
                    completion.saturating_add(charge.service_cost())
                })
        }
    }

    fn group(index: usize) -> PdGroupId {
        PdGroupId::parse(format!("group-{index}")).unwrap()
    }

    fn prefill(index: usize, generation: u128) -> PrefillId {
        let origin = format!("http://prefill-{index}.test:30000");
        PrefillId::new(
            HttpOrigin::parse(&origin).unwrap(),
            Uuid::from_u128(generation),
        )
        .unwrap()
    }

    fn candidate(index: usize, available: bool, service_cost_ms: u64) -> PdPrefillCandidate {
        PdPrefillCandidate::new(
            PdPrefillRoute::new(group(index), prefill(index, (index + 1) as u128)),
            index,
            available,
            Duration::from_millis(service_cost_ms),
        )
    }

    #[test]
    fn arbitrary_pool_cardinality_uses_stable_manifest_order() {
        let scheduler = PdPrefillScheduler::new();
        let candidates = (0..11)
            .rev()
            .map(|index| candidate(index, true, 10))
            .collect::<Vec<_>>();
        let mut reservations = Vec::new();

        for request_index in 0..11 {
            reservations.push(
                scheduler
                    .reserve(
                        format!("request-{request_index}"),
                        &candidates,
                        &SerialCostModel,
                    )
                    .unwrap(),
            );
        }

        assert_eq!(
            reservations
                .iter()
                .map(|reservation| reservation.route().group_id().as_str().to_string())
                .collect::<Vec<_>>(),
            (0..11)
                .map(|index| format!("group-{index}"))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unavailable_groups_are_excluded_before_prediction() {
        let scheduler = PdPrefillScheduler::new();
        let candidates = vec![
            candidate(0, false, 1),
            candidate(1, true, 50),
            candidate(2, false, 1),
        ];

        let reservation = scheduler
            .reserve("available-only", &candidates, &SerialCostModel)
            .unwrap();

        assert_eq!(reservation.route().group_id(), &group(1));
        assert_eq!(
            reservation.predicted_completion(),
            Duration::from_millis(50)
        );
    }

    #[test]
    fn concurrent_selection_and_charging_are_atomic() {
        let scheduler = Arc::new(PdPrefillScheduler::new());
        let candidates = Arc::new(
            (0..8)
                .map(|index| candidate(index, true, 10))
                .collect::<Vec<_>>(),
        );
        let barrier = Arc::new(Barrier::new(65));
        let mut handles = Vec::new();
        for request_index in 0..64 {
            let scheduler = Arc::clone(&scheduler);
            let candidates = Arc::clone(&candidates);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                scheduler
                    .reserve(
                        format!("concurrent-{request_index}"),
                        &candidates,
                        &SerialCostModel,
                    )
                    .unwrap()
            }));
        }
        barrier.wait();
        let reservations = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        let mut counts = HashMap::new();
        for reservation in &reservations {
            *counts
                .entry(reservation.route().group_id().clone())
                .or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 8);
        assert!(counts.values().all(|count| *count == 8));
        assert_eq!(
            (0..8)
                .map(|index| scheduler.outstanding_reservations(&group(index)))
                .sum::<usize>(),
            64
        );

        drop(reservations);
        assert!((0..8).all(|index| scheduler.outstanding_reservations(&group(index)) == 0));
    }

    #[test]
    fn terminal_ownership_releases_only_its_exact_group_and_generation() {
        let scheduler = PdPrefillScheduler::new();
        let initial_candidates = vec![candidate(0, true, 10), candidate(1, true, 20)];
        let first = scheduler
            .reserve("first", &initial_candidates, &SerialCostModel)
            .unwrap();
        let replacement_prefill = prefill(0, 100);
        let replacement_candidates = vec![
            PdPrefillCandidate::new(
                PdPrefillRoute::new(group(0), replacement_prefill.clone()),
                0,
                true,
                Duration::from_millis(10),
            ),
            candidate(1, true, 20),
        ];
        let second = scheduler
            .reserve("second", &replacement_candidates, &SerialCostModel)
            .unwrap();
        assert_eq!(first.route().group_id(), &group(0));
        assert_eq!(second.route().group_id(), &group(0));
        assert_eq!(first.route().prefill_id(), &prefill(0, 1));
        assert_eq!(second.route().prefill_id(), &replacement_prefill);
        assert_eq!(scheduler.outstanding_reservations(&group(0)), 2);

        first.complete();
        assert_eq!(scheduler.outstanding_reservations(&group(0)), 1);
        assert_eq!(scheduler.outstanding_reservations(&group(1)), 0);
        second.cancel();
        assert_eq!(scheduler.outstanding_reservations(&group(0)), 0);
    }

    #[test]
    fn duplicate_logical_request_cannot_own_two_groups() {
        let scheduler = PdPrefillScheduler::new();
        let candidates = vec![candidate(0, true, 10), candidate(1, true, 10)];
        let reservation = scheduler
            .reserve("same-request", &candidates, &SerialCostModel)
            .unwrap();

        assert!(matches!(
            scheduler.reserve("same-request", &candidates, &SerialCostModel),
            Err(PdPrefillSchedulerError::DuplicateLogicalRequest(request_id))
                if request_id == "same-request"
        ));

        drop(reservation);
        assert!(scheduler
            .reserve("same-request", &candidates, &SerialCostModel)
            .is_ok());
    }

    #[test]
    fn malformed_candidate_sets_fail_before_reservation() {
        let scheduler = PdPrefillScheduler::new();
        let duplicate_group = vec![
            candidate(0, true, 10),
            PdPrefillCandidate::new(
                PdPrefillRoute::new(group(0), prefill(0, 2)),
                1,
                true,
                Duration::from_millis(10),
            ),
        ];
        assert!(matches!(
            scheduler.reserve("duplicate-group", &duplicate_group, &SerialCostModel),
            Err(PdPrefillSchedulerError::DuplicateGroupCandidate(group_id))
                if group_id == group(0)
        ));

        let duplicate_index = vec![
            candidate(0, true, 10),
            PdPrefillCandidate::new(
                PdPrefillRoute::new(group(1), prefill(1, 2)),
                0,
                true,
                Duration::from_millis(10),
            ),
        ];
        assert!(matches!(
            scheduler.reserve("duplicate-index", &duplicate_index, &SerialCostModel),
            Err(PdPrefillSchedulerError::DuplicateManifestIndex(0))
        ));
    }
}
