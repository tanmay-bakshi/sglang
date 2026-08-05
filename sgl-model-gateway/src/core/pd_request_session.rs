//! Cancellation-safe ownership for one disaggregated prefill/decode request.

use std::{fmt, future::pending, sync::Arc, time::Duration};

use bytes::Bytes;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use super::{
    pd_decoder_directory::{
        DecoderDirectoryEntry, PdDirectoryError, PdProcessDirectory, PrefillDirectoryEntry,
    },
    pd_decoder_grant::{
        BindReconciliationGrant, BoundPreparedGrant, DecoderControlAuthorization,
        DecoderGrantControlClient, DecoderRequestTemplate, EngineGrantError, PrefillId,
        PromotionReconciliationGrant, RetainedEngineGrant, UnboundPreparedGrant,
    },
    pd_decoder_pool::{
        DecoderAssignmentCohort, DecoderAssignmentReconciliationError, DecoderPoolError,
        LogicalRequestOwner, PendingAdmission, PendingAdmissionDisposition,
        PendingReconciliationError, PendingReserveOutcome, PendingSchedulingCharge,
        RetryDisposition,
    },
    retry::BackoffCalculator,
    Worker,
};
use crate::config::types::RetryConfig;

const PREPARED_GRANT_TTL: Duration = Duration::from_secs(30);
const DROPPED_SESSION_REASON: &str = "request_session_dropped";
const ESTABLISHMENT_CANCELLED_REASON: &str = "request_establishment_cancelled";
const ESTABLISHMENT_FAILED_REASON: &str = "request_establishment_failed";

type SessionResult = Result<(), PdRequestSessionError>;

/// A ready disaggregated request whose lifecycle authority remains actor-owned.
#[must_use = "a promoted PD request session must be completed or aborted"]
pub struct PdRequestSession {
    prefill_worker: Arc<dyn Worker>,
    decoder_worker: Arc<dyn Worker>,
    request_body: Bytes,
    command_tx: Option<mpsc::UnboundedSender<SessionCommand>>,
    terminal_command_sent: bool,
}

impl fmt::Debug for PdRequestSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PdRequestSession")
            .field("prefill_url", &self.prefill_worker.url())
            .field("decoder_url", &self.decoder_worker.url())
            .field("request_body_bytes", &self.request_body.len())
            .field("terminal_command_sent", &self.terminal_command_sent)
            .finish_non_exhaustive()
    }
}

impl PdRequestSession {
    /// Establish one promoted decoder assignment before inference dispatch.
    ///
    /// The spawned owner starts before directory admission. Cancellation of
    /// this future therefore closes the command channel but cannot destroy any
    /// authority already acquired by the actor.
    pub async fn establish(
        directory: Arc<PdProcessDirectory>,
        selected_prefill: &PrefillId,
        request_id: impl Into<String>,
        model_id: Option<&str>,
        template: DecoderRequestTemplate,
        control: &DecoderGrantControlClient,
        retry_config: &RetryConfig,
    ) -> Result<Self, PdRequestSessionError> {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = oneshot::channel();
        let inputs = ActorInputs {
            directory,
            selected_prefill: selected_prefill.clone(),
            request_id: request_id.into(),
            model_id: model_id.map(str::to_string),
            template,
            control: control.clone(),
            retry_config: retry_config.clone(),
        };
        tokio::spawn(run_request_actor(inputs, command_rx, ready_tx));
        let ready = ready_rx
            .await
            .map_err(|_| PdRequestSessionError::ActorUnavailable)??;
        Ok(Self {
            prefill_worker: ready.prefill_worker,
            decoder_worker: ready.decoder_worker,
            request_body: ready.request_body,
            command_tx: Some(command_tx),
            terminal_command_sent: false,
        })
    }

    /// Selected prefill worker generation.
    pub fn prefill_worker(&self) -> &Arc<dyn Worker> {
        &self.prefill_worker
    }

    /// Pool-selected decoder worker generation.
    pub fn decoder_worker(&self) -> &Arc<dyn Worker> {
        &self.decoder_worker
    }

    /// Exact bound inference bytes shared by both dispatches.
    pub fn request_body(&self) -> Bytes {
        self.request_body.clone()
    }

    /// Reconcile successful engine release before relinquishing ownership.
    ///
    /// Dropping the returned future cannot stop reconciliation because the
    /// detached owner has already accepted the terminal command.
    pub async fn complete(mut self) -> SessionResult {
        let (response_tx, response_rx) = oneshot::channel();
        self.send_terminal_command(SessionCommand::Complete(response_tx))?;
        response_rx
            .await
            .map_err(|_| PdRequestSessionError::ActorUnavailable)?
    }

    /// Reconcile engine abort before relinquishing ownership.
    ///
    /// Dropping the returned future cannot stop reconciliation because the
    /// detached owner has already accepted the terminal command.
    pub async fn abort(mut self, reason_code: impl Into<String>) -> SessionResult {
        let (response_tx, response_rx) = oneshot::channel();
        self.send_terminal_command(SessionCommand::Abort {
            reason_code: reason_code.into(),
            response: Some(response_tx),
        })?;
        response_rx
            .await
            .map_err(|_| PdRequestSessionError::ActorUnavailable)?
    }

    fn send_terminal_command(
        &mut self,
        command: SessionCommand,
    ) -> Result<(), PdRequestSessionError> {
        let command_tx = self
            .command_tx
            .as_ref()
            .ok_or(PdRequestSessionError::ActorUnavailable)?;
        command_tx
            .send(command)
            .map_err(|_| PdRequestSessionError::ActorUnavailable)?;
        self.terminal_command_sent = true;
        Ok(())
    }
}

impl Drop for PdRequestSession {
    fn drop(&mut self) {
        if self.terminal_command_sent {
            return;
        }
        self.command_tx.take();
    }
}

enum SessionCommand {
    Complete(oneshot::Sender<SessionResult>),
    Abort {
        reason_code: String,
        response: Option<oneshot::Sender<SessionResult>>,
    },
}

struct ActorInputs {
    directory: Arc<PdProcessDirectory>,
    selected_prefill: PrefillId,
    request_id: String,
    model_id: Option<String>,
    template: DecoderRequestTemplate,
    control: DecoderGrantControlClient,
    retry_config: RetryConfig,
}

struct SessionReady {
    prefill_worker: Arc<dyn Worker>,
    decoder_worker: Arc<dyn Worker>,
    request_body: Bytes,
}

enum SessionAuthority {
    Idle,
    Pending {
        pending: PendingAdmission,
        reserve_started: bool,
    },
    Unbound {
        pending: PendingAdmission,
        grant: UnboundPreparedGrant,
    },
    Binding {
        pending: PendingAdmission,
        grant: BindReconciliationGrant,
    },
    Bound {
        pending: PendingAdmission,
        grant: BoundPreparedGrant,
    },
    Reserved {
        cohort: DecoderAssignmentCohort,
    },
    Promoting {
        cohort: DecoderAssignmentCohort,
        grant: PromotionReconciliationGrant,
    },
    Active {
        cohort: DecoderAssignmentCohort,
        grant: RetainedEngineGrant,
    },
    Quarantined {
        cohort: DecoderAssignmentCohort,
    },
    Terminal,
    Transitioning,
}

impl SessionAuthority {
    fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Pending {
                reserve_started: false,
                ..
            } => "pending-unpolled",
            Self::Pending {
                reserve_started: true,
                ..
            } => "reserve-ambiguous",
            Self::Unbound { .. } => "unbound",
            Self::Binding { .. } => "binding",
            Self::Bound { .. } => "bound",
            Self::Reserved { .. } => "reserved",
            Self::Promoting { .. } => "promoting",
            Self::Active { .. } => "active",
            Self::Quarantined { .. } => "quarantined",
            Self::Terminal => "terminal",
            Self::Transitioning => "transitioning",
        }
    }
}

struct PdRequestActor {
    directory: Arc<PdProcessDirectory>,
    prefill: Arc<PrefillDirectoryEntry>,
    decoder: Option<Arc<DecoderDirectoryEntry>>,
    owner: Option<LogicalRequestOwner>,
    template: DecoderRequestTemplate,
    control: DecoderGrantControlClient,
    retry_config: RetryConfig,
    request_body: Option<Bytes>,
    authority: SessionAuthority,
}

impl PdRequestActor {
    fn new(
        inputs: ActorInputs,
        prefill: Arc<PrefillDirectoryEntry>,
        owner: LogicalRequestOwner,
    ) -> Self {
        Self {
            directory: inputs.directory,
            prefill,
            decoder: None,
            owner: Some(owner),
            template: inputs.template,
            control: inputs.control,
            retry_config: inputs.retry_config,
            request_body: None,
            authority: SessionAuthority::Idle,
        }
    }

    async fn establish(
        &mut self,
        commands: &mut mpsc::UnboundedReceiver<SessionCommand>,
    ) -> Result<Option<SessionReady>, PdRequestSessionError> {
        if commands.is_closed() {
            self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON)
                .await?;
            return Ok(None);
        }

        let allocator_attempts = self.retry_config.max_retries.max(1);
        for allocator_attempt in 0..allocator_attempts {
            if let Err(error) = self.begin_admission() {
                return Err(self.cleanup_establishment_failure(error).await);
            }
            let authorization = match self.decoder_authorization() {
                Ok(authorization) => authorization,
                Err(error) => return Err(self.cleanup_establishment_failure(error).await),
            };
            let reserve_outcome = match self.reconcile_reserve(authorization, commands).await {
                Ok(Some(outcome)) => outcome,
                Ok(None) => return Ok(None),
                Err(error) => return Err(self.cleanup_establishment_failure(error).await),
            };
            match reserve_outcome {
                PendingReserveOutcome::Prepared(grant) => {
                    if let Err(error) = self.install_unbound(*grant) {
                        return Err(self.cleanup_establishment_failure(error).await);
                    }
                }
                PendingReserveOutcome::Refused(disposition) => {
                    self.release_resolved_pending()?;
                    if disposition == PendingAdmissionDisposition::Terminal
                        || allocator_attempt + 1 == allocator_attempts
                    {
                        let error = PdRequestSessionError::AllocatorRefused(disposition);
                        return Err(self.cleanup_establishment_failure(error).await);
                    }
                    continue;
                }
            }

            match self.bind(commands).await {
                Ok(true) => {}
                Ok(false) => return Ok(None),
                Err(error) => return Err(self.cleanup_establishment_failure(error).await),
            }
            if let Err(error) = self.install_cohort() {
                return Err(self.cleanup_establishment_failure(error).await);
            }
            match self.promote(commands).await {
                Ok(true) => {}
                Ok(false) => return Ok(None),
                Err(error) => return Err(self.cleanup_establishment_failure(error).await),
            }
            return match self.ready() {
                Ok(ready) => Ok(Some(ready)),
                Err(error) => Err(self.cleanup_establishment_failure(error).await),
            };
        }
        unreachable!("allocator attempt loop returns on every terminal path")
    }

    fn begin_admission(&mut self) -> Result<(), PdRequestSessionError> {
        let owner = self
            .owner
            .as_ref()
            .ok_or(PdRequestSessionError::NotActive)?;
        let charge = PendingSchedulingCharge::new(self.template.child_count(), 0, 0)?;
        let pending = self.directory.begin_admission(
            self.prefill.id(),
            owner,
            &self.template,
            PREPARED_GRANT_TTL,
            charge,
        )?;
        let decoder = self
            .directory
            .decoder(pending.decoder_id())
            .ok_or(PdRequestSessionError::SelectedDecoderMissing)?;
        self.decoder = Some(decoder);
        self.authority = SessionAuthority::Pending {
            pending,
            reserve_started: false,
        };
        Ok(())
    }

    fn decoder_authorization(&self) -> Result<DecoderControlAuthorization, PdRequestSessionError> {
        let api_key = self
            .decoder
            .as_ref()
            .ok_or(PdRequestSessionError::SelectedDecoderMissing)?
            .worker()
            .api_key()
            .as_ref()
            .ok_or(PdRequestSessionError::DecoderApiKeyMissing)?;
        DecoderControlAuthorization::new(api_key.clone()).map_err(Into::into)
    }

    async fn reconcile_reserve(
        &mut self,
        authorization: DecoderControlAuthorization,
        commands: &mut mpsc::UnboundedReceiver<SessionCommand>,
    ) -> Result<Option<PendingReserveOutcome>, PdRequestSessionError> {
        let reservation_attempt_id = match &self.authority {
            SessionAuthority::Pending { pending, .. } => pending.reservation_attempt_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut reserve = match &mut self.authority {
            SessionAuthority::Pending {
                pending,
                reserve_started,
            } => {
                let reserve = pending.begin_reserve(&self.control, authorization)?;
                *reserve_started = true;
                reserve
            }
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut attempt = 0;
        loop {
            let result = tokio::select! {
                biased;
                command = commands.recv() => {
                    reject_unready_command(command);
                    drop(reserve);
                    self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON).await?;
                    return Ok(None);
                }
                result = reserve.reconcile_reserve() => result,
            };
            match result {
                Ok(outcome) => return Ok(Some(outcome)),
                Err(error) => {
                    warn!(
                        reservation_attempt_id = %reservation_attempt_id,
                        error = %error,
                        "PD reserve reconciliation remains pending"
                    );
                    if retry_delay_or_cancel(&self.retry_config, &mut attempt, commands).await {
                        drop(reserve);
                        self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON)
                            .await?;
                        return Ok(None);
                    }
                }
            }
        }
    }

    fn install_unbound(
        &mut self,
        grant: UnboundPreparedGrant,
    ) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Pending { pending, .. } = authority else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        self.authority = SessionAuthority::Unbound { pending, grant };
        Ok(())
    }

    async fn bind(
        &mut self,
        commands: &mut mpsc::UnboundedReceiver<SessionCommand>,
    ) -> Result<bool, PdRequestSessionError> {
        self.begin_bind()?;
        let attempts = self.retry_config.max_retries.max(1);
        let mut retry_attempt = 0;
        for attempt in 0..attempts {
            let grant_id = match &self.authority {
                SessionAuthority::Binding { grant, .. } => grant.grant_id(),
                _ => return Err(PdRequestSessionError::NotActive),
            };
            let result = tokio::select! {
                biased;
                command = commands.recv() => {
                    reject_unready_command(command);
                    self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON).await?;
                    return Ok(false);
                }
                result = async {
                    match &mut self.authority {
                        SessionAuthority::Binding { grant, .. } => {
                            grant.reconcile_bind().await
                        }
                        _ => Err(EngineGrantError::ProtocolViolation(
                            "PD request actor lost bind authority".to_string(),
                        )),
                    }
                } => result,
            };
            match result {
                Ok(bound) => {
                    self.install_bound(bound)?;
                    return Ok(true);
                }
                Err(error) => {
                    warn!(
                        grant_id = %grant_id,
                        error = %error,
                        "PD bind reconciliation remains ambiguous"
                    );
                    if attempt + 1 == attempts {
                        return Err(error.into());
                    }
                    if retry_delay_or_cancel(&self.retry_config, &mut retry_attempt, commands).await
                    {
                        self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON)
                            .await?;
                        return Ok(false);
                    }
                }
            }
        }
        unreachable!("bind attempt loop returns on every terminal path")
    }

    fn begin_bind(&mut self) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Unbound { pending, mut grant } = authority else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        match grant.begin_bind() {
            Ok(binding) => {
                self.authority = SessionAuthority::Binding {
                    pending,
                    grant: binding,
                };
                Ok(())
            }
            Err(error) => {
                self.authority = SessionAuthority::Unbound { pending, grant };
                Err(error.into())
            }
        }
    }

    fn install_bound(&mut self, bound: BoundPreparedGrant) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Binding { pending, .. } = authority else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        self.authority = SessionAuthority::Bound {
            pending,
            grant: bound,
        };
        Ok(())
    }

    fn install_cohort(&mut self) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Bound {
            mut pending,
            mut grant,
        } = authority
        else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        let request_body = grant.request_body();
        match self.prefill.pool().bind_grant(&mut pending, &mut grant) {
            Ok(cohort) => {
                self.request_body = Some(request_body);
                self.authority = SessionAuthority::Reserved { cohort };
                Ok(())
            }
            Err(error) => {
                self.authority = SessionAuthority::Bound { pending, grant };
                Err(error.into())
            }
        }
    }

    async fn promote(
        &mut self,
        commands: &mut mpsc::UnboundedReceiver<SessionCommand>,
    ) -> Result<bool, PdRequestSessionError> {
        self.begin_promotion()?;
        let attempts = self.retry_config.max_retries.max(1);
        let mut retry_attempt = 0;
        for attempt in 0..attempts {
            let grant_id = match &self.authority {
                SessionAuthority::Promoting { grant, .. } => grant.grant_id(),
                _ => return Err(PdRequestSessionError::NotActive),
            };
            let result = tokio::select! {
                biased;
                command = commands.recv() => {
                    reject_unready_command(command);
                    self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON).await?;
                    return Ok(false);
                }
                result = async {
                    match &mut self.authority {
                        SessionAuthority::Promoting { grant, .. } => {
                            grant.reconcile_promotion().await
                        }
                        _ => Err(EngineGrantError::ProtocolViolation(
                            "PD request actor lost promotion authority".to_string(),
                        )),
                    }
                } => result,
            };
            match result {
                Ok(retained) => {
                    self.install_retained(retained)?;
                    return Ok(true);
                }
                Err(error) => {
                    warn!(
                        grant_id = %grant_id,
                        error = %error,
                        "PD promotion reconciliation remains ambiguous"
                    );
                    if attempt + 1 == attempts {
                        return Err(error.into());
                    }
                    if retry_delay_or_cancel(&self.retry_config, &mut retry_attempt, commands).await
                    {
                        self.cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON)
                            .await?;
                        return Ok(false);
                    }
                }
            }
        }
        unreachable!("promotion attempt loop returns on every terminal path")
    }

    fn begin_promotion(&mut self) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Reserved { mut cohort } = authority else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        match self.prefill.pool().begin_promotion(&mut cohort) {
            Ok(grant) => {
                self.authority = SessionAuthority::Promoting { cohort, grant };
                Ok(())
            }
            Err(error) => {
                self.authority = SessionAuthority::Reserved { cohort };
                Err(error.into())
            }
        }
    }

    fn install_retained(
        &mut self,
        retained: RetainedEngineGrant,
    ) -> Result<(), PdRequestSessionError> {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Promoting { cohort, .. } = authority else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        self.authority = SessionAuthority::Active {
            cohort,
            grant: retained,
        };
        Ok(())
    }

    fn ready(&self) -> Result<SessionReady, PdRequestSessionError> {
        if !matches!(self.authority, SessionAuthority::Active { .. }) {
            return Err(PdRequestSessionError::NotActive);
        }
        let decoder_worker = self
            .decoder
            .as_ref()
            .map(|decoder| Arc::clone(decoder.worker()))
            .ok_or(PdRequestSessionError::SelectedDecoderMissing)?;
        let request_body = self
            .request_body
            .clone()
            .ok_or(PdRequestSessionError::NotActive)?;
        Ok(SessionReady {
            prefill_worker: Arc::clone(self.prefill.worker()),
            decoder_worker,
            request_body,
        })
    }

    async fn complete_and_finalize(&mut self) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let assignment_id = match &self.authority {
            SessionAuthority::Active { cohort, .. } => cohort.assignment_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut completion = match &mut self.authority {
            SessionAuthority::Active { cohort, grant } => pool.begin_completion(cohort, grant)?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut attempt = 0;
        loop {
            match completion.reconcile().await {
                Ok(()) => break,
                Err(error) => {
                    warn!(
                        assignment_id = %assignment_id,
                        error = %error,
                        "PD completion reconciliation remains pending"
                    );
                    sleep_before_retry(&self.retry_config, &mut attempt).await;
                }
            }
        }
        drop(completion);
        self.authority = SessionAuthority::Terminal;
        self.finalize_owner()
    }

    async fn cancel_and_finalize(&mut self, reason_code: &str) -> SessionResult {
        loop {
            match &self.authority {
                SessionAuthority::Idle => {
                    self.authority = SessionAuthority::Terminal;
                    return self.finalize_owner();
                }
                SessionAuthority::Pending {
                    reserve_started: false,
                    ..
                } => {
                    self.drop_unpolled_pending()?;
                }
                SessionAuthority::Pending {
                    reserve_started: true,
                    ..
                } => {
                    self.resolve_ambiguous_reserve().await?;
                }
                SessionAuthority::Unbound { .. } => {
                    self.cancel_unbound_pending().await?;
                }
                SessionAuthority::Binding { .. } => {
                    self.cancel_binding_pending().await?;
                }
                SessionAuthority::Bound { .. } => {
                    self.cancel_bound_pending().await?;
                }
                SessionAuthority::Reserved { .. } => {
                    self.cancel_reserved().await?;
                }
                SessionAuthority::Promoting { .. } => {
                    self.abort_promoting(reason_code).await?;
                }
                SessionAuthority::Active { .. } => {
                    self.abort_active(reason_code).await?;
                }
                SessionAuthority::Quarantined { cohort } => {
                    return Err(PdRequestSessionError::Quarantined(cohort.assignment_id()));
                }
                SessionAuthority::Terminal => return self.finalize_owner(),
                SessionAuthority::Transitioning => {
                    return Err(PdRequestSessionError::InternalTransition)
                }
            }
        }
    }

    fn drop_unpolled_pending(&mut self) -> SessionResult {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        let SessionAuthority::Pending {
            pending,
            reserve_started: false,
        } = authority
        else {
            self.authority = authority;
            return Err(PdRequestSessionError::NotActive);
        };
        drop(pending);
        self.authority = SessionAuthority::Idle;
        Ok(())
    }

    async fn resolve_ambiguous_reserve(&mut self) -> SessionResult {
        let reservation_attempt_id = match &self.authority {
            SessionAuthority::Pending { pending, .. } => pending.reservation_attempt_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let decoder_id = match &self.authority {
            SessionAuthority::Pending { pending, .. } => pending.decoder_id().clone(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut reserve = match &mut self.authority {
            SessionAuthority::Pending {
                pending,
                reserve_started: true,
            } => pending.resume_reserve()?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut attempt = 0;
        let outcome = loop {
            match reserve.reconcile_reserve().await {
                Ok(outcome) => break outcome,
                Err(error) => {
                    let availability = self.directory.decoder_availability(&decoder_id);
                    warn!(
                        reservation_attempt_id = %reservation_attempt_id,
                        decoder_id = %decoder_id,
                        ?availability,
                        error = %error,
                        "Cancelled PD request is retaining exact ambiguous reserve authority"
                    );
                    sleep_before_retry(&self.retry_config, &mut attempt).await;
                }
            }
        };
        drop(reserve);
        match outcome {
            PendingReserveOutcome::Prepared(grant) => self.install_unbound(*grant),
            PendingReserveOutcome::Refused(_) => self.release_resolved_pending(),
        }
    }

    async fn cancel_unbound_pending(&mut self) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let reservation_attempt_id = match &self.authority {
            SessionAuthority::Unbound { pending, .. } => pending.reservation_attempt_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut cancellation = match &mut self.authority {
            SessionAuthority::Unbound { pending, grant } => pool
                .begin_unbound_pending_cancellation(
                    pending,
                    grant,
                    PendingAdmissionDisposition::Terminal,
                )?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        reconcile_pending_cancellation(
            &mut cancellation,
            reservation_attempt_id,
            &self.retry_config,
        )
        .await;
        drop(cancellation);
        self.release_resolved_pending()
    }

    async fn cancel_binding_pending(&mut self) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let reservation_attempt_id = match &self.authority {
            SessionAuthority::Binding { pending, .. } => pending.reservation_attempt_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut cancellation = match &mut self.authority {
            SessionAuthority::Binding { pending, grant } => pool.begin_bind_pending_cancellation(
                pending,
                grant,
                PendingAdmissionDisposition::Terminal,
            )?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        reconcile_pending_cancellation(
            &mut cancellation,
            reservation_attempt_id,
            &self.retry_config,
        )
        .await;
        drop(cancellation);
        self.release_resolved_pending()
    }

    async fn cancel_bound_pending(&mut self) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let reservation_attempt_id = match &self.authority {
            SessionAuthority::Bound { pending, .. } => pending.reservation_attempt_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut cancellation = match &mut self.authority {
            SessionAuthority::Bound { pending, grant } => pool.begin_bound_pending_cancellation(
                pending,
                grant,
                PendingAdmissionDisposition::Terminal,
            )?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        reconcile_pending_cancellation(
            &mut cancellation,
            reservation_attempt_id,
            &self.retry_config,
        )
        .await;
        drop(cancellation);
        self.release_resolved_pending()
    }

    fn release_resolved_pending(&mut self) -> SessionResult {
        let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
        match authority {
            SessionAuthority::Pending { pending, .. }
            | SessionAuthority::Unbound { pending, .. }
            | SessionAuthority::Binding { pending, .. }
            | SessionAuthority::Bound { pending, .. } => {
                drop(pending);
                self.authority = SessionAuthority::Idle;
                Ok(())
            }
            authority => {
                self.authority = authority;
                Err(PdRequestSessionError::NotActive)
            }
        }
    }

    async fn cancel_reserved(&mut self) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let assignment_id = match &self.authority {
            SessionAuthority::Reserved { cohort } => cohort.assignment_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut cancellation = match &mut self.authority {
            SessionAuthority::Reserved { cohort } => {
                pool.begin_cancellation(cohort, RetryDisposition::Terminal)?
            }
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut attempt = 0;
        loop {
            match cancellation.reconcile().await {
                Ok(()) => break,
                Err(error) => {
                    warn!(
                        assignment_id = %assignment_id,
                        error = %error,
                        "PD reserved-cohort cancellation remains pending"
                    );
                    sleep_before_retry(&self.retry_config, &mut attempt).await;
                }
            }
        }
        drop(cancellation);
        self.authority = SessionAuthority::Terminal;
        self.finalize_owner()
    }

    async fn abort_promoting(&mut self, reason_code: &str) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let assignment_id = match &self.authority {
            SessionAuthority::Promoting { cohort, .. } => cohort.assignment_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut abort = match &mut self.authority {
            SessionAuthority::Promoting { cohort, grant } => pool.begin_abort_from_promotion(
                cohort,
                grant,
                reason_code,
                None,
                RetryDisposition::Terminal,
            )?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        reconcile_abort(&mut abort, assignment_id, &self.retry_config).await;
        drop(abort);
        self.finish_abort(pool, assignment_id)
    }

    async fn abort_active(&mut self, reason_code: &str) -> SessionResult {
        let pool = self.prefill.pool().clone();
        let assignment_id = match &self.authority {
            SessionAuthority::Active { cohort, .. } => cohort.assignment_id(),
            _ => return Err(PdRequestSessionError::NotActive),
        };
        let mut abort = match &mut self.authority {
            SessionAuthority::Active { cohort, grant } => pool.begin_abort_from_retained(
                cohort,
                grant,
                reason_code,
                None,
                RetryDisposition::Terminal,
            )?,
            _ => return Err(PdRequestSessionError::NotActive),
        };
        reconcile_abort(&mut abort, assignment_id, &self.retry_config).await;
        drop(abort);
        self.finish_abort(pool, assignment_id)
    }

    fn finish_abort(
        &mut self,
        pool: super::pd_decoder_pool::DecoderPool,
        assignment_id: uuid::Uuid,
    ) -> SessionResult {
        let quarantined = match &self.authority {
            SessionAuthority::Promoting { cohort, .. }
            | SessionAuthority::Active { cohort, .. } => {
                pool.cohort_remains_quarantined(cohort)?
            }
            _ => return Err(PdRequestSessionError::NotActive),
        };
        if quarantined {
            let authority = std::mem::replace(&mut self.authority, SessionAuthority::Transitioning);
            let cohort = match authority {
                SessionAuthority::Promoting { cohort, .. }
                | SessionAuthority::Active { cohort, .. } => cohort,
                authority => {
                    self.authority = authority;
                    return Err(PdRequestSessionError::NotActive);
                }
            };
            self.authority = SessionAuthority::Quarantined { cohort };
            return Err(PdRequestSessionError::Quarantined(assignment_id));
        }
        self.authority = SessionAuthority::Terminal;
        self.finalize_owner()
    }

    fn finalize_owner(&mut self) -> SessionResult {
        let owner = self
            .owner
            .as_mut()
            .ok_or(PdRequestSessionError::NotActive)?;
        self.prefill.pool().finalize_request(owner)?;
        self.owner = None;
        Ok(())
    }

    async fn cleanup_establishment_failure(
        &mut self,
        primary: PdRequestSessionError,
    ) -> PdRequestSessionError {
        match self.cancel_and_finalize(ESTABLISHMENT_FAILED_REASON).await {
            Ok(()) => primary,
            Err(cleanup) => {
                warn!(
                    primary_error = %primary,
                    cleanup_error = %cleanup,
                    "PD request establishment failed and retained unresolved authority"
                );
                cleanup
            }
        }
    }

    fn is_fully_finalized(&self) -> bool {
        self.owner.is_none() && matches!(self.authority, SessionAuthority::Terminal)
    }

    async fn retain_unresolved(self, context: &'static str) {
        warn!(
            request_id = ?self.owner.as_ref().map(LogicalRequestOwner::request_id),
            authority = self.authority.name(),
            context,
            "PD request actor is retaining unresolved authority for process lifetime"
        );
        pending::<()>().await;
    }
}

async fn run_request_actor(
    inputs: ActorInputs,
    mut commands: mpsc::UnboundedReceiver<SessionCommand>,
    ready_tx: oneshot::Sender<Result<SessionReady, PdRequestSessionError>>,
) {
    if commands.is_closed() {
        return;
    }
    let request = inputs
        .directory
        .begin_prefill_request(&inputs.selected_prefill, inputs.request_id.clone());
    let (prefill, owner) = match request {
        Ok(request) => request,
        Err(error) => {
            let _ = ready_tx.send(Err(error.into()));
            return;
        }
    };
    let model_mismatch = inputs
        .model_id
        .as_deref()
        .is_some_and(|model_id| !prefill.worker().supports_model(model_id));
    let mut actor = PdRequestActor::new(inputs, prefill, owner);
    if model_mismatch {
        let error = match actor.cancel_and_finalize(ESTABLISHMENT_FAILED_REASON).await {
            Ok(()) => PdRequestSessionError::PrefillModelMismatch,
            Err(error) => error,
        };
        let retain = !actor.is_fully_finalized();
        let _ = ready_tx.send(Err(error));
        if retain {
            actor.retain_unresolved("prefill model mismatch").await;
        }
        return;
    }

    match actor.establish(&mut commands).await {
        Ok(Some(ready)) => {
            if ready_tx.send(Ok(ready)).is_err() {
                let result = actor
                    .cancel_and_finalize(ESTABLISHMENT_CANCELLED_REASON)
                    .await;
                if let Err(error) = result {
                    warn!(
                        error = %error,
                        "PD request readiness receiver disappeared during actor cleanup"
                    );
                }
                if !actor.is_fully_finalized() {
                    actor
                        .retain_unresolved("readiness receiver disappeared")
                        .await;
                }
                return;
            }
            serve_ready_session(actor, commands).await;
        }
        Ok(None) => {
            if !actor.is_fully_finalized() {
                actor
                    .retain_unresolved("establishment caller disappeared")
                    .await;
            }
        }
        Err(error) => {
            let retain = !actor.is_fully_finalized();
            let _ = ready_tx.send(Err(error));
            if retain {
                actor.retain_unresolved("establishment failure").await;
            }
        }
    }
}

async fn serve_ready_session(
    mut actor: PdRequestActor,
    mut commands: mpsc::UnboundedReceiver<SessionCommand>,
) {
    let command = commands.recv().await;
    let (result, response) = match command {
        Some(SessionCommand::Complete(response)) => {
            (actor.complete_and_finalize().await, Some(response))
        }
        Some(SessionCommand::Abort {
            reason_code,
            response,
        }) => (actor.cancel_and_finalize(&reason_code).await, response),
        None => (
            actor.cancel_and_finalize(DROPPED_SESSION_REASON).await,
            None,
        ),
    };
    let retain = !actor.is_fully_finalized();
    if let Some(response) = response {
        let _ = response.send(result);
    } else if let Err(error) = result {
        warn!(
            error = %error,
            "Dropped PD request session could not release all authority"
        );
    }
    if retain {
        actor.retain_unresolved("terminal reconciliation").await;
    }
}

fn reject_unready_command(command: Option<SessionCommand>) {
    let error = || Err(PdRequestSessionError::NotActive);
    match command {
        Some(SessionCommand::Complete(response)) => {
            let _ = response.send(error());
        }
        Some(SessionCommand::Abort {
            response: Some(response),
            ..
        }) => {
            let _ = response.send(error());
        }
        Some(SessionCommand::Abort { response: None, .. }) => {}
        None => {}
    }
}

async fn reconcile_pending_cancellation(
    cancellation: &mut super::pd_decoder_pool::PendingCancellationReconciliation<'_>,
    reservation_attempt_id: uuid::Uuid,
    retry_config: &RetryConfig,
) {
    let mut attempt = 0;
    loop {
        match cancellation.reconcile_cancellation().await {
            Ok(_) => return,
            Err(error) => {
                warn!(
                    reservation_attempt_id = %reservation_attempt_id,
                    error = %error,
                    "PD pending-admission cancellation remains pending"
                );
                sleep_before_retry(retry_config, &mut attempt).await;
            }
        }
    }
}

async fn reconcile_abort(
    abort: &mut super::pd_decoder_pool::PoolAbortReconciliation<'_>,
    assignment_id: uuid::Uuid,
    retry_config: &RetryConfig,
) {
    let mut attempt = 0;
    loop {
        match abort.reconcile().await {
            Ok(()) => return,
            Err(error) => {
                warn!(
                    assignment_id = %assignment_id,
                    error = %error,
                    "PD abort reconciliation remains pending"
                );
                sleep_before_retry(retry_config, &mut attempt).await;
            }
        }
    }
}

async fn retry_delay_or_cancel(
    retry_config: &RetryConfig,
    attempt: &mut u32,
    commands: &mut mpsc::UnboundedReceiver<SessionCommand>,
) -> bool {
    let delay = BackoffCalculator::calculate_delay(retry_config, *attempt);
    *attempt = (*attempt).saturating_add(1);
    tokio::select! {
        biased;
        command = commands.recv() => {
            reject_unready_command(command);
            true
        }
        _ = tokio::time::sleep(delay) => false,
    }
}

async fn sleep_before_retry(retry_config: &RetryConfig, attempt: &mut u32) {
    let delay = BackoffCalculator::calculate_delay(retry_config, *attempt);
    *attempt = (*attempt).saturating_add(1);
    tokio::time::sleep(delay).await;
}

/// Failure to establish or terminalize one exact PD request session.
#[derive(Debug, Error)]
pub enum PdRequestSessionError {
    #[error("selected prefill does not serve the requested model")]
    PrefillModelMismatch,
    #[error("decoder selected by admission is no longer present in the process directory")]
    SelectedDecoderMissing,
    #[error("selected decoder has no configured control-plane API key")]
    DecoderApiKeyMissing,
    #[error("decoder allocator authoritatively refused admission: {0:?}")]
    AllocatorRefused(PendingAdmissionDisposition),
    #[error("PD request session actor is unavailable")]
    ActorUnavailable,
    #[error("PD request session is not active")]
    NotActive,
    #[error("PD request actor entered an invalid authority transition")]
    InternalTransition,
    #[error("decoder assignment {0} remains authoritatively quarantined")]
    Quarantined(uuid::Uuid),
    #[error(transparent)]
    Directory(#[from] PdDirectoryError),
    #[error(transparent)]
    Pool(#[from] DecoderPoolError),
    #[error(transparent)]
    Engine(#[from] EngineGrantError),
    #[error(transparent)]
    Pending(#[from] PendingReconciliationError),
    #[error(transparent)]
    Assignment(#[from] DecoderAssignmentReconciliationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    const API_KEY_SECRET: &str = "session-worker-api-key";
    const REQUEST_BODY_SECRET: &str = "session-request-body-secret";

    fn session() -> (PdRequestSession, mpsc::UnboundedReceiver<SessionCommand>) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let prefill_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill.test")
                .api_key(API_KEY_SECRET)
                .build(),
        );
        let decoder_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://decoder.test")
                .api_key(API_KEY_SECRET)
                .build(),
        );
        (
            PdRequestSession {
                prefill_worker,
                decoder_worker,
                request_body: Bytes::from_static(REQUEST_BODY_SECRET.as_bytes()),
                command_tx: Some(command_tx),
                terminal_command_sent: false,
            },
            command_rx,
        )
    }

    #[test]
    fn debug_omits_request_and_worker_credentials() {
        let (session, _commands) = session();
        let debug = format!("{session:?}");

        assert!(!debug.contains(API_KEY_SECRET));
        assert!(!debug.contains(REQUEST_BODY_SECRET));
        assert!(debug.contains("request_body_bytes"));
    }

    #[tokio::test]
    async fn dropping_ready_handle_closes_the_actor_command_channel() {
        let (session, mut commands) = session();

        drop(session);

        assert!(commands.recv().await.is_none());
    }

    #[tokio::test]
    async fn accepted_completion_command_outlives_cancelled_waiter() {
        let (session, mut commands) = session();
        let waiter = tokio::spawn(session.complete());

        let command = commands.recv().await.unwrap();
        let SessionCommand::Complete(response) = command else {
            panic!("completion emitted the wrong terminal command");
        };

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(response);
    }
}
