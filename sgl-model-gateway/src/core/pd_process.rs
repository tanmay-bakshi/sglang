//! Typed process-generation capabilities for disaggregated inference.

use std::{net::Ipv4Addr, num::NonZeroUsize, sync::Arc};

use thiserror::Error;
use url::Host;
use uuid::Uuid;

/// Versioned schema for process-generation metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PdMetadataSchema {
    V1,
}

/// Role of one disaggregated inference process generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PdProcessRole {
    Prefill,
    Decode,
}

/// KV-transfer wire protocol implemented by a process generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvTransferProtocol {
    PackedV4,
}

impl KvTransferProtocol {
    /// Return the canonical protocol identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PackedV4 => "packed-v4",
        }
    }
}

/// Prepared decoder-grant control protocol implemented by a process generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PreparedGrantProtocol {
    V1,
}

impl PreparedGrantProtocol {
    /// Return the canonical protocol identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "control-v1",
        }
    }
}

/// Explicit non-local authority used to bootstrap prefill-to-decode transfer.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefillBootstrapEndpoint {
    host: Arc<str>,
    port: u16,
}

impl PrefillBootstrapEndpoint {
    /// Construct and validate a bootstrap authority without DNS resolution.
    pub fn new(host: impl Into<String>, port: u16) -> Result<Self, PdProcessMetadataError> {
        let host = host.into();
        if host.is_empty() || host.trim() != host || host.chars().any(char::is_control) || port == 0
        {
            return Err(PdProcessMetadataError::InvalidBootstrapAuthority);
        }

        let parsed =
            Host::parse(&host).map_err(|_| PdProcessMetadataError::InvalidBootstrapAuthority)?;
        match parsed {
            Host::Domain(domain) if is_localhost_name(&domain) => {
                return Err(PdProcessMetadataError::InvalidBootstrapAuthority);
            }
            Host::Ipv4(address) if is_unusable_ipv4(address) => {
                return Err(PdProcessMetadataError::InvalidBootstrapAuthority);
            }
            Host::Ipv6(address)
                if address.is_loopback() || address.is_unspecified() || address.is_multicast() =>
            {
                return Err(PdProcessMetadataError::InvalidBootstrapAuthority);
            }
            Host::Ipv6(address) if address.to_ipv4_mapped().is_some_and(is_unusable_ipv4) => {
                return Err(PdProcessMetadataError::InvalidBootstrapAuthority);
            }
            Host::Domain(_) | Host::Ipv4(_) | Host::Ipv6(_) => {}
        }

        Ok(Self {
            host: Arc::from(host),
            port,
        })
    }

    /// Return the validated host component.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Return the nonzero port.
    pub fn port(&self) -> u16 {
        self.port
    }
}

fn is_localhost_name(domain: &str) -> bool {
    let domain = domain.trim_end_matches('.');
    domain.eq_ignore_ascii_case("localhost") || domain.to_ascii_lowercase().ends_with(".localhost")
}

fn is_unusable_ipv4(address: Ipv4Addr) -> bool {
    address.is_loopback() || address.is_unspecified() || address.is_multicast()
}

/// Closed compatibility and launch metadata for one process generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdProcessMetadata {
    schema: PdMetadataSchema,
    launch_instance_id: Uuid,
    role: PdProcessRole,
    tensor_parallel_size: NonZeroUsize,
    data_parallel_size: NonZeroUsize,
    model_fingerprint: Arc<str>,
    logical_kv_layout_fingerprint: Arc<str>,
    kv_dtype: Arc<str>,
    page_size: NonZeroUsize,
    kv_transfer_protocol: KvTransferProtocol,
    prepared_grant_protocol: PreparedGrantProtocol,
    prefill_bootstrap_endpoint: Option<PrefillBootstrapEndpoint>,
}

impl PdProcessMetadata {
    /// Construct validated process-generation metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        schema: PdMetadataSchema,
        launch_instance_id: Uuid,
        role: PdProcessRole,
        tensor_parallel_size: usize,
        data_parallel_size: usize,
        model_fingerprint: impl Into<String>,
        logical_kv_layout_fingerprint: impl Into<String>,
        kv_dtype: impl Into<String>,
        page_size: usize,
        kv_transfer_protocol: KvTransferProtocol,
        prepared_grant_protocol: PreparedGrantProtocol,
        prefill_bootstrap_endpoint: Option<PrefillBootstrapEndpoint>,
    ) -> Result<Self, PdProcessMetadataError> {
        if launch_instance_id.is_nil() {
            return Err(PdProcessMetadataError::NilLaunchInstance);
        }
        let tensor_parallel_size = NonZeroUsize::new(tensor_parallel_size)
            .ok_or(PdProcessMetadataError::InvalidTensorParallelSize)?;
        let data_parallel_size = NonZeroUsize::new(data_parallel_size)
            .ok_or(PdProcessMetadataError::InvalidDataParallelSize)?;
        if data_parallel_size.get() != 1 {
            return Err(PdProcessMetadataError::UnsupportedDataParallelSize(
                data_parallel_size.get(),
            ));
        }

        match (role, prefill_bootstrap_endpoint.is_some()) {
            (PdProcessRole::Prefill, false) => {
                return Err(PdProcessMetadataError::MissingPrefillBootstrap);
            }
            (PdProcessRole::Decode, true) => {
                return Err(PdProcessMetadataError::UnexpectedDecodeBootstrap);
            }
            _ => {}
        }

        Ok(Self {
            schema,
            launch_instance_id,
            role,
            tensor_parallel_size,
            data_parallel_size,
            model_fingerprint: canonical_digest("model", model_fingerprint.into())?,
            logical_kv_layout_fingerprint: canonical_digest(
                "logical KV layout",
                logical_kv_layout_fingerprint.into(),
            )?,
            kv_dtype: canonical_token(kv_dtype.into())?,
            page_size: NonZeroUsize::new(page_size)
                .ok_or(PdProcessMetadataError::InvalidPageSize)?,
            kv_transfer_protocol,
            prepared_grant_protocol,
            prefill_bootstrap_endpoint,
        })
    }

    /// Return the metadata schema version.
    pub fn schema(&self) -> PdMetadataSchema {
        self.schema
    }

    /// Return the launch-generation UUID.
    pub fn launch_instance_id(&self) -> Uuid {
        self.launch_instance_id
    }

    /// Return the process role.
    pub fn role(&self) -> PdProcessRole {
        self.role
    }

    /// Return the physical tensor-parallel width.
    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size.get()
    }

    /// Return the physical data-parallel width.
    pub fn data_parallel_size(&self) -> usize {
        self.data_parallel_size.get()
    }

    /// Return the model-weight fingerprint.
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    /// Return the TP-independent logical KV-layout fingerprint.
    pub fn logical_kv_layout_fingerprint(&self) -> &str {
        &self.logical_kv_layout_fingerprint
    }

    /// Return the canonical KV-cache dtype.
    pub fn kv_dtype(&self) -> &str {
        &self.kv_dtype
    }

    /// Return the KV-cache page size.
    pub fn page_size(&self) -> usize {
        self.page_size.get()
    }

    /// Return the KV-transfer wire protocol.
    pub fn kv_transfer_protocol(&self) -> KvTransferProtocol {
        self.kv_transfer_protocol
    }

    /// Return the prepared-grant control protocol.
    pub fn prepared_grant_protocol(&self) -> PreparedGrantProtocol {
        self.prepared_grant_protocol
    }

    /// Return the prefill bootstrap endpoint when the role requires one.
    pub fn prefill_bootstrap_endpoint(&self) -> Option<&PrefillBootstrapEndpoint> {
        self.prefill_bootstrap_endpoint.as_ref()
    }

    /// Compare every semantic and protocol field independent of physical TP.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.schema == other.schema
            && self.data_parallel_size == other.data_parallel_size
            && self.model_fingerprint == other.model_fingerprint
            && self.logical_kv_layout_fingerprint == other.logical_kv_layout_fingerprint
            && self.kv_dtype == other.kv_dtype
            && self.page_size == other.page_size
            && self.kv_transfer_protocol == other.kv_transfer_protocol
            && self.prepared_grant_protocol == other.prepared_grant_protocol
    }
}

fn canonical_digest(name: &'static str, value: String) -> Result<Arc<str>, PdProcessMetadataError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PdProcessMetadataError::InvalidFingerprint(name));
    }
    Ok(Arc::from(value))
}

fn canonical_token(value: String) -> Result<Arc<str>, PdProcessMetadataError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(PdProcessMetadataError::InvalidKvDtype);
    }
    Ok(Arc::from(value))
}

/// Validation failures for typed process-generation metadata.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PdProcessMetadataError {
    #[error("PD launch instance cannot be nil")]
    NilLaunchInstance,
    #[error("PD tensor parallel size must be nonzero")]
    InvalidTensorParallelSize,
    #[error("PD data parallel size must be nonzero")]
    InvalidDataParallelSize,
    #[error("PD directory supports only DP1, received DP{0}")]
    UnsupportedDataParallelSize(usize),
    #[error("prefill metadata requires an explicit bootstrap endpoint")]
    MissingPrefillBootstrap,
    #[error("decode metadata cannot advertise a prefill bootstrap endpoint")]
    UnexpectedDecodeBootstrap,
    #[error("prefill bootstrap authority must be explicit, non-local, and usable")]
    InvalidBootstrapAuthority,
    #[error("{0} fingerprint must be a canonical lowercase 256-bit hex digest")]
    InvalidFingerprint(&'static str),
    #[error("KV dtype must be a canonical lowercase token")]
    InvalidKvDtype,
    #[error("KV page size must be nonzero")]
    InvalidPageSize,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn bootstrap_authority_rejects_local_and_unusable_hosts() {
        for host in [
            "",
            " localhost",
            "localhost",
            "localhost.",
            "api.localhost",
            "api.localhost.",
            "127.0.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "[::1]",
            "[::]",
            "[ff02::1]",
            "[::ffff:127.0.0.1]",
            "[::ffff:0.0.0.0]",
            "[::ffff:224.0.0.1]",
            "https://bootstrap.test",
            "bootstrap.test:5000",
        ] {
            assert!(
                PrefillBootstrapEndpoint::new(host, 50_051).is_err(),
                "accepted {host:?}"
            );
        }
        assert!(PrefillBootstrapEndpoint::new("bootstrap.test", 0).is_err());
        assert!(PrefillBootstrapEndpoint::new("bootstrap.test", 50_051).is_ok());
        assert!(PrefillBootstrapEndpoint::new("10.20.30.40", 50_051).is_ok());
        assert!(PrefillBootstrapEndpoint::new("[2001:db8::1]", 50_051).is_ok());
    }

    #[test]
    fn metadata_closes_role_dp_and_fingerprint_shapes() {
        let endpoint = PrefillBootstrapEndpoint::new("bootstrap.test", 50_051).unwrap();
        let valid = PdProcessMetadata::new(
            PdMetadataSchema::V1,
            Uuid::new_v4(),
            PdProcessRole::Prefill,
            4,
            1,
            DIGEST_A,
            DIGEST_B,
            "bf16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            Some(endpoint.clone()),
        )
        .unwrap();
        assert_eq!(valid.tensor_parallel_size(), 4);
        assert_eq!(valid.prefill_bootstrap_endpoint(), Some(&endpoint));

        assert!(PdProcessMetadata::new(
            PdMetadataSchema::V1,
            Uuid::new_v4(),
            PdProcessRole::Prefill,
            4,
            2,
            DIGEST_A,
            DIGEST_B,
            "bf16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            Some(endpoint.clone()),
        )
        .is_err());
        assert!(PdProcessMetadata::new(
            PdMetadataSchema::V1,
            Uuid::new_v4(),
            PdProcessRole::Decode,
            1,
            1,
            "not-a-digest",
            DIGEST_B,
            "BF16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            None,
        )
        .is_err());
        assert!(PdProcessMetadata::new(
            PdMetadataSchema::V1,
            Uuid::new_v4(),
            PdProcessRole::Decode,
            1,
            1,
            DIGEST_A,
            DIGEST_B,
            "bf16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            Some(endpoint),
        )
        .is_err());
    }
}
