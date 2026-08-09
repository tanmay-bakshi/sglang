//! Immutable deployment topology for strict prefill-decode routing.

use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{HttpOrigin, PdProcessMetadata, PdProcessRole, PrefillBootstrapEndpoint};

/// Schema version for a strict prefill-decode topology document.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PdTopologySchema {
    #[serde(rename = "pd-topology-v1")]
    V1,
}

/// Canonical identifier for one independently scheduled prefill group.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(try_from = "String", into = "String")]
pub struct PdGroupId(String);

impl PdGroupId {
    /// Parse a canonical group identifier.
    pub fn parse(value: impl Into<String>) -> Result<Self, PdTopologyError> {
        let value = value.into();
        let mut chars = value.chars();
        let starts_with_letter = chars
            .next()
            .is_some_and(|character| character.is_ascii_lowercase());
        let remaining_is_canonical = chars.all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '-'
                || character == '_'
        });
        if value.len() > 64 || !starts_with_letter || !remaining_is_canonical {
            return Err(PdTopologyError::InvalidGroupId(value));
        }
        Ok(Self(value))
    }

    /// Return the canonical identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for PdGroupId {
    type Error = PdTopologyError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<PdGroupId> for String {
    fn from(value: PdGroupId) -> Self {
        value.0
    }
}

/// Bootstrap authority frozen for one topology prefill.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdTopologyBootstrapEndpoint {
    pub host: String,
    pub port: u16,
}

impl PdTopologyBootstrapEndpoint {
    /// Convert the document value through the process-metadata validator.
    pub fn validate(&self) -> Result<PrefillBootstrapEndpoint, PdTopologyError> {
        PrefillBootstrapEndpoint::new(self.host.clone(), self.port)
            .map_err(|_| PdTopologyError::InvalidBootstrapEndpoint)
    }
}

/// Frozen prefill process specification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdPrefillSpec {
    pub origin: HttpOrigin,
    pub tensor_parallel_size: usize,
    pub bootstrap_endpoint: PdTopologyBootstrapEndpoint,
}

/// Frozen decoder process specification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdDecoderSpec {
    pub origin: HttpOrigin,
    pub tensor_parallel_size: usize,
}

/// One static prefill and its exclusively owned decoder processes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdTopologyGroup {
    pub id: PdGroupId,
    pub prefill: PdPrefillSpec,
    pub decoders: Vec<PdDecoderSpec>,
}

/// Complete immutable prefill-decode deployment topology.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdTopology {
    pub schema: PdTopologySchema,
    pub groups: Vec<PdTopologyGroup>,
}

#[derive(Serialize)]
struct CanonicalPdTopology<'a> {
    schema: &'static str,
    groups: Vec<CanonicalPdTopologyGroup<'a>>,
}

#[derive(Serialize)]
struct CanonicalPdTopologyGroup<'a> {
    id: &'a str,
    prefill: CanonicalPdPrefillSpec<'a>,
    decoders: Vec<CanonicalPdDecoderSpec<'a>>,
}

#[derive(Serialize)]
struct CanonicalPdPrefillSpec<'a> {
    origin: &'a str,
    tensor_parallel_size: usize,
    bootstrap_endpoint: CanonicalPdBootstrapEndpoint<'a>,
}

#[derive(Serialize)]
struct CanonicalPdBootstrapEndpoint<'a> {
    host: &'a str,
    port: u16,
}

#[derive(Serialize)]
struct CanonicalPdDecoderSpec<'a> {
    origin: &'a str,
    tensor_parallel_size: usize,
}

impl<'a> From<&'a PdTopology> for CanonicalPdTopology<'a> {
    fn from(topology: &'a PdTopology) -> Self {
        Self {
            schema: match topology.schema {
                PdTopologySchema::V1 => "pd-topology-v1",
            },
            groups: topology
                .groups
                .iter()
                .map(|group| CanonicalPdTopologyGroup {
                    id: group.id.as_str(),
                    prefill: CanonicalPdPrefillSpec {
                        origin: group.prefill.origin.as_str(),
                        tensor_parallel_size: group.prefill.tensor_parallel_size,
                        bootstrap_endpoint: CanonicalPdBootstrapEndpoint {
                            host: &group.prefill.bootstrap_endpoint.host,
                            port: group.prefill.bootstrap_endpoint.port,
                        },
                    },
                    decoders: group
                        .decoders
                        .iter()
                        .map(|decoder| CanonicalPdDecoderSpec {
                            origin: decoder.origin.as_str(),
                            tensor_parallel_size: decoder.tensor_parallel_size,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl PdTopology {
    /// Parse and fully validate a topology JSON document.
    pub fn from_json(json: &str) -> Result<Self, PdTopologyError> {
        let topology: Self = serde_json::from_str(json)?;
        topology.validate()?;
        Ok(topology)
    }

    /// Read, parse, and fully validate a topology JSON document.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, PdTopologyError> {
        let json = fs::read_to_string(path).map_err(PdTopologyError::Read)?;
        Self::from_json(&json)
    }

    /// Validate all cross-object invariants.
    pub fn validate(&self) -> Result<(), PdTopologyError> {
        if self.groups.is_empty() {
            return Err(PdTopologyError::NoGroups);
        }

        let mut group_ids = HashSet::new();
        let mut origins = HashSet::new();
        for group in &self.groups {
            if !group_ids.insert(group.id.clone()) {
                return Err(PdTopologyError::DuplicateGroupId(
                    group.id.as_str().to_string(),
                ));
            }
            if !matches!(group.prefill.tensor_parallel_size, 1 | 2 | 4) {
                return Err(PdTopologyError::UnsupportedPrefillTensorParallelSize {
                    origin: group.prefill.origin.to_string(),
                    tensor_parallel_size: group.prefill.tensor_parallel_size,
                });
            }
            group.prefill.bootstrap_endpoint.validate()?;
            if !origins.insert(group.prefill.origin.clone()) {
                return Err(PdTopologyError::DuplicateOrigin(
                    group.prefill.origin.to_string(),
                ));
            }
            if group.decoders.is_empty() {
                return Err(PdTopologyError::NoDecoders {
                    group_id: group.id.as_str().to_string(),
                });
            }
            for decoder in &group.decoders {
                if !matches!(decoder.tensor_parallel_size, 1 | 2) {
                    return Err(PdTopologyError::UnsupportedDecodeTensorParallelSize {
                        origin: decoder.origin.to_string(),
                        tensor_parallel_size: decoder.tensor_parallel_size,
                    });
                }
                if group.prefill.tensor_parallel_size % decoder.tensor_parallel_size != 0 {
                    return Err(PdTopologyError::IncompatibleTensorParallelSizes {
                        group_id: group.id.as_str().to_string(),
                        prefill_tensor_parallel_size: group.prefill.tensor_parallel_size,
                        decode_tensor_parallel_size: decoder.tensor_parallel_size,
                    });
                }
                if !origins.insert(decoder.origin.clone()) {
                    return Err(PdTopologyError::DuplicateOrigin(decoder.origin.to_string()));
                }
            }
        }
        Ok(())
    }

    /// Return the expected static process specification for an origin.
    pub fn process_spec(&self, origin: &HttpOrigin) -> Option<PdTopologyProcessSpec<'_>> {
        let group = self.group_for_origin(origin)?;
        if &group.prefill.origin == origin {
            return Some(PdTopologyProcessSpec::Prefill {
                group_id: &group.id,
                spec: &group.prefill,
            });
        }
        group
            .decoders
            .iter()
            .find(|decoder| &decoder.origin == origin)
            .map(|spec| PdTopologyProcessSpec::Decoder {
                group_id: &group.id,
                spec,
            })
    }

    /// Return the manifest group that owns a process origin.
    pub fn group_for_origin(&self, origin: &HttpOrigin) -> Option<&PdTopologyGroup> {
        for group in &self.groups {
            if &group.prefill.origin == origin {
                return Some(group);
            }
            if group
                .decoders
                .iter()
                .any(|decoder| &decoder.origin == origin)
            {
                return Some(group);
            }
        }
        None
    }

    /// Match authenticated process metadata against its frozen static specification.
    pub fn match_registration(
        &self,
        origin: &HttpOrigin,
        metadata: &PdProcessMetadata,
    ) -> Result<&PdGroupId, PdTopologyRegistrationError> {
        match self.process_spec(origin) {
            Some(PdTopologyProcessSpec::Prefill { group_id, spec }) => {
                if metadata.role() != PdProcessRole::Prefill {
                    return Err(PdTopologyRegistrationError::RoleMismatch);
                }
                if metadata.tensor_parallel_size() != spec.tensor_parallel_size {
                    return Err(PdTopologyRegistrationError::TensorParallelMismatch {
                        expected: spec.tensor_parallel_size,
                        actual: metadata.tensor_parallel_size(),
                    });
                }
                let expected = spec
                    .bootstrap_endpoint
                    .validate()
                    .expect("validated topology retains a valid bootstrap endpoint");
                if metadata.prefill_bootstrap_endpoint() != Some(&expected) {
                    return Err(PdTopologyRegistrationError::BootstrapMismatch);
                }
                Ok(group_id)
            }
            Some(PdTopologyProcessSpec::Decoder { group_id, spec }) => {
                if metadata.role() != PdProcessRole::Decode {
                    return Err(PdTopologyRegistrationError::RoleMismatch);
                }
                if metadata.tensor_parallel_size() != spec.tensor_parallel_size {
                    return Err(PdTopologyRegistrationError::TensorParallelMismatch {
                        expected: spec.tensor_parallel_size,
                        actual: metadata.tensor_parallel_size(),
                    });
                }
                if metadata.prefill_bootstrap_endpoint().is_some() {
                    return Err(PdTopologyRegistrationError::BootstrapMismatch);
                }
                Ok(group_id)
            }
            None => Err(PdTopologyRegistrationError::UnmanifestedOrigin),
        }
    }

    /// Serialize the topology under the cross-language canonical byte contract.
    ///
    /// Object fields are emitted in schema/group/spec declaration order, arrays retain
    /// manifest order, origins use their normalized spelling, strings use compact JSON
    /// escaping with non-ASCII text encoded directly as UTF-8, and the byte sequence has
    /// no BOM, insignificant whitespace, or trailing newline.
    pub fn canonical_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&CanonicalPdTopology::from(self))
            .expect("a validated topology contains only infallibly serializable values")
    }

    /// Return the canonical SHA-256 digest of the validated document.
    pub fn sha256(&self) -> String {
        format!("{:x}", Sha256::digest(self.canonical_json_bytes()))
    }

    /// Return process origins in manifest order.
    pub fn origins(&self) -> impl Iterator<Item = &HttpOrigin> {
        self.groups.iter().flat_map(|group| {
            std::iter::once(&group.prefill.origin)
                .chain(group.decoders.iter().map(|decoder| &decoder.origin))
        })
    }
}

/// Borrowed static process specification returned by topology lookup.
#[derive(Clone, Copy, Debug)]
pub enum PdTopologyProcessSpec<'a> {
    Prefill {
        group_id: &'a PdGroupId,
        spec: &'a PdPrefillSpec,
    },
    Decoder {
        group_id: &'a PdGroupId,
        spec: &'a PdDecoderSpec,
    },
}

/// Invalid topology document.
#[derive(Debug, Error)]
pub enum PdTopologyError {
    #[error("failed to read PD topology: {0}")]
    Read(std::io::Error),
    #[error("invalid PD topology JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("PD topology must contain at least one group")]
    NoGroups,
    #[error("invalid PD topology group id {0:?}")]
    InvalidGroupId(String),
    #[error("duplicate PD topology group id {0:?}")]
    DuplicateGroupId(String),
    #[error("PD topology group {group_id:?} must contain at least one decoder")]
    NoDecoders { group_id: String },
    #[error(
        "unsupported prefill tensor_parallel_size {tensor_parallel_size} for {origin}; expected 1, 2, or 4"
    )]
    UnsupportedPrefillTensorParallelSize {
        origin: String,
        tensor_parallel_size: usize,
    },
    #[error(
        "unsupported decode tensor_parallel_size {tensor_parallel_size} for {origin}; expected 1 or 2"
    )]
    UnsupportedDecodeTensorParallelSize {
        origin: String,
        tensor_parallel_size: usize,
    },
    #[error(
        "PD topology group {group_id:?} has incompatible tensor-parallel sizes: prefill TP{prefill_tensor_parallel_size}, decode TP{decode_tensor_parallel_size}"
    )]
    IncompatibleTensorParallelSizes {
        group_id: String,
        prefill_tensor_parallel_size: usize,
        decode_tensor_parallel_size: usize,
    },
    #[error("duplicate PD process origin {0}")]
    DuplicateOrigin(String),
    #[error("invalid prefill bootstrap endpoint")]
    InvalidBootstrapEndpoint,
}

/// Process registration that violates a frozen topology.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum PdTopologyRegistrationError {
    #[error("PD process origin is absent from the frozen topology")]
    UnmanifestedOrigin,
    #[error("PD process role differs from the frozen topology")]
    RoleMismatch,
    #[error("PD process tensor parallel size differs from the frozen topology: expected {expected}, observed {actual}")]
    TensorParallelMismatch { expected: usize, actual: usize },
    #[error("PD process bootstrap endpoint differs from the frozen topology")]
    BootstrapMismatch,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::core::{KvTransferProtocol, PdMetadataSchema, PreparedGrantProtocol};

    fn topology_json() -> String {
        json!({
            "schema": "pd-topology-v1",
            "groups": [
                {
                    "id": "g0",
                    "prefill": {
                        "origin": "http://127.0.0.1:32100",
                        "tensor_parallel_size": 2,
                        "bootstrap_endpoint": {"host": "gemma-dev-1", "port": 32150}
                    },
                    "decoders": [
                        {"origin": "http://127.0.0.1:32101", "tensor_parallel_size": 1}
                    ]
                },
                {
                    "id": "g1",
                    "prefill": {
                        "origin": "http://127.0.0.1:32102",
                        "tensor_parallel_size": 1,
                        "bootstrap_endpoint": {"host": "gemma-dev-1", "port": 32151}
                    },
                    "decoders": [
                        {"origin": "http://127.0.0.1:32103", "tensor_parallel_size": 1},
                        {"origin": "http://127.0.0.1:32104", "tensor_parallel_size": 1}
                    ]
                }
            ]
        })
        .to_string()
    }

    fn metadata(
        role: PdProcessRole,
        tensor_parallel_size: usize,
        bootstrap: Option<PrefillBootstrapEndpoint>,
    ) -> PdProcessMetadata {
        PdProcessMetadata::new(
            PdMetadataSchema::V1,
            Uuid::new_v4(),
            role,
            tensor_parallel_size,
            1,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "fp8".to_string(),
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            bootstrap,
        )
        .unwrap()
    }

    #[test]
    fn parses_canonical_multi_group_topology() {
        let topology = PdTopology::from_json(&topology_json()).unwrap();

        assert_eq!(topology.groups.len(), 2);
        assert_eq!(topology.groups[1].prefill.tensor_parallel_size, 1);
        assert_eq!(topology.groups[1].decoders.len(), 2);
        assert_eq!(topology.sha256().len(), 64);
        assert_eq!(topology.origins().collect::<HashSet<_>>().len(), 5);
    }

    #[test]
    fn digest_is_stable_across_json_formatting_and_origin_spelling() {
        let first = PdTopology::from_json(&topology_json()).unwrap();
        let second = PdTopology::from_json(
            &topology_json().replace("http://127.0.0.1:32100", "HTTP://127.0.0.1:32100/"),
        )
        .unwrap();

        assert_eq!(first.sha256(), second.sha256());
    }

    #[test]
    fn canonical_bytes_match_cross_language_test_vector() {
        let input = r#"{
            "groups": [
                {
                    "decoders": [
                        {"tensor_parallel_size": 1, "origin": "http://127.0.0.1:32101"},
                        {"tensor_parallel_size": 1, "origin": "http://127.0.0.1:32102"},
                        {"tensor_parallel_size": 1, "origin": "http://127.0.0.1:32103"}
                    ],
                    "prefill": {
                        "bootstrap_endpoint": {"port": 32150, "host": "gemma-dev-1"},
                        "tensor_parallel_size": 1,
                        "origin": "HTTP://127.0.0.1:32100/"
                    },
                    "id": "group-0"
                },
                {
                    "decoders": [
                        {"origin": "http://127.0.0.1:32105", "tensor_parallel_size": 1},
                        {"origin": "http://127.0.0.1:32106", "tensor_parallel_size": 1},
                        {"origin": "http://127.0.0.1:32107", "tensor_parallel_size": 1}
                    ],
                    "prefill": {
                        "origin": "http://127.0.0.1:32104",
                        "tensor_parallel_size": 1,
                        "bootstrap_endpoint": {"host": "gemma-dev-1", "port": 32151}
                    },
                    "id": "group-1"
                }
            ],
            "schema": "pd-topology-v1"
        }"#;
        let expected = concat!(
            r#"{"schema":"pd-topology-v1","groups":[{"id":"group-0","prefill":{"origin":"http://127.0.0.1:32100","tensor_parallel_size":1,"bootstrap_endpoint":{"host":"gemma-dev-1","port":32150}},"decoders":["#,
            r#"{"origin":"http://127.0.0.1:32101","tensor_parallel_size":1},{"origin":"http://127.0.0.1:32102","tensor_parallel_size":1},{"origin":"http://127.0.0.1:32103","tensor_parallel_size":1}]},"#,
            r#"{"id":"group-1","prefill":{"origin":"http://127.0.0.1:32104","tensor_parallel_size":1,"bootstrap_endpoint":{"host":"gemma-dev-1","port":32151}},"decoders":["#,
            r#"{"origin":"http://127.0.0.1:32105","tensor_parallel_size":1},{"origin":"http://127.0.0.1:32106","tensor_parallel_size":1},{"origin":"http://127.0.0.1:32107","tensor_parallel_size":1}]}]}"#,
        );
        let topology = PdTopology::from_json(input).unwrap();

        assert_eq!(topology.canonical_json_bytes(), expected.as_bytes());
        assert_eq!(
            topology.sha256(),
            "24afeedde0264f2874a77f18266ad57b096e397c43bc65e16bd46334b52df5a2"
        );
    }

    #[test]
    fn rejects_unknown_fields_and_invalid_cross_object_state() {
        let mut document: serde_json::Value = serde_json::from_str(&topology_json()).unwrap();
        document["mystery"] = json!(true);
        assert!(PdTopology::from_json(&document.to_string()).is_err());

        let mut duplicate: serde_json::Value = serde_json::from_str(&topology_json()).unwrap();
        duplicate["groups"][1]["decoders"][0]["origin"] =
            duplicate["groups"][0]["prefill"]["origin"].clone();
        assert!(matches!(
            PdTopology::from_json(&duplicate.to_string()),
            Err(PdTopologyError::DuplicateOrigin(_))
        ));

        let mut no_decoders: serde_json::Value = serde_json::from_str(&topology_json()).unwrap();
        no_decoders["groups"][0]["decoders"] = json!([]);
        assert!(matches!(
            PdTopology::from_json(&no_decoders.to_string()),
            Err(PdTopologyError::NoDecoders { .. })
        ));

        let mut unsupported_prefill: serde_json::Value =
            serde_json::from_str(&topology_json()).unwrap();
        unsupported_prefill["groups"][0]["prefill"]["tensor_parallel_size"] = json!(3);
        assert!(matches!(
            PdTopology::from_json(&unsupported_prefill.to_string()),
            Err(PdTopologyError::UnsupportedPrefillTensorParallelSize { .. })
        ));

        let mut incompatible_decode: serde_json::Value =
            serde_json::from_str(&topology_json()).unwrap();
        incompatible_decode["groups"][1]["prefill"]["tensor_parallel_size"] = json!(1);
        incompatible_decode["groups"][1]["decoders"][0]["tensor_parallel_size"] = json!(2);
        assert!(matches!(
            PdTopology::from_json(&incompatible_decode.to_string()),
            Err(PdTopologyError::IncompatibleTensorParallelSizes { .. })
        ));
    }

    #[test]
    fn matches_static_registration_without_freezing_generation() {
        let topology = PdTopology::from_json(&topology_json()).unwrap();
        let origin = HttpOrigin::parse("http://127.0.0.1:32100").unwrap();
        let first = metadata(
            PdProcessRole::Prefill,
            2,
            Some(PrefillBootstrapEndpoint::new("gemma-dev-1", 32150).unwrap()),
        );
        let second = metadata(
            PdProcessRole::Prefill,
            2,
            Some(PrefillBootstrapEndpoint::new("gemma-dev-1", 32150).unwrap()),
        );

        assert_ne!(first.launch_instance_id(), second.launch_instance_id());
        assert_eq!(
            topology
                .match_registration(&origin, &first)
                .unwrap()
                .as_str(),
            "g0"
        );
        assert_eq!(
            topology
                .match_registration(&origin, &second)
                .unwrap()
                .as_str(),
            "g0"
        );
    }

    #[test]
    fn rejects_registration_metadata_drift() {
        let topology = PdTopology::from_json(&topology_json()).unwrap();
        let origin = HttpOrigin::parse("http://127.0.0.1:32100").unwrap();

        assert_eq!(
            topology
                .match_registration(&origin, &metadata(PdProcessRole::Decode, 2, None))
                .unwrap_err(),
            PdTopologyRegistrationError::RoleMismatch
        );
        assert!(matches!(
            topology.match_registration(
                &origin,
                &metadata(
                    PdProcessRole::Prefill,
                    4,
                    Some(PrefillBootstrapEndpoint::new("gemma-dev-1", 32150).unwrap()),
                ),
            ),
            Err(PdTopologyRegistrationError::TensorParallelMismatch { .. })
        ));
        assert_eq!(
            topology
                .match_registration(
                    &origin,
                    &metadata(
                        PdProcessRole::Prefill,
                        2,
                        Some(PrefillBootstrapEndpoint::new("gemma-dev-1", 32152).unwrap()),
                    ),
                )
                .unwrap_err(),
            PdTopologyRegistrationError::BootstrapMismatch
        );
    }
}
