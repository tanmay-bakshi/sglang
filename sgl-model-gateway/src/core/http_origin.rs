//! Canonical HTTP process origins.

use std::{fmt, sync::Arc};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use url::Url;

/// Canonical identity and endpoint authority for one HTTP process.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HttpOrigin {
    canonical: Arc<str>,
    host: Arc<str>,
}

impl HttpOrigin {
    /// Parse a fully qualified HTTP(S) origin.
    pub fn parse(value: &str) -> Result<Self, HttpOriginError> {
        let scheme_prefix_len = if value
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        {
            7
        } else if value
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
        {
            8
        } else {
            return Err(HttpOriginError::InvalidSyntax);
        };
        if value.trim() != value || value.chars().any(char::is_control) || value.contains('\\') {
            return Err(HttpOriginError::InvalidSyntax);
        }
        let authority_and_suffix = &value[scheme_prefix_len..];
        let authority_len = authority_and_suffix
            .find(['/', '?', '#'])
            .unwrap_or(authority_and_suffix.len());
        if authority_len == 0 {
            return Err(HttpOriginError::MissingHost);
        }
        let authority = &authority_and_suffix[..authority_len];
        if authority.contains('@') {
            return Err(HttpOriginError::CredentialsNotAllowed);
        }
        if authority.ends_with(':') {
            return Err(HttpOriginError::InvalidSyntax);
        }

        let parsed =
            Url::parse(value).map_err(|error| HttpOriginError::InvalidUrl(error.to_string()))?;
        let host = parsed.host_str().ok_or(HttpOriginError::MissingHost)?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(HttpOriginError::CredentialsNotAllowed);
        }
        if parsed.query().is_some() {
            return Err(HttpOriginError::QueryNotAllowed);
        }
        if parsed.fragment().is_some() {
            return Err(HttpOriginError::FragmentNotAllowed);
        }
        let raw_path = &authority_and_suffix[authority_len..];
        if !raw_path.is_empty() && raw_path != "/" {
            return Err(HttpOriginError::PathNotAllowed);
        }
        if parsed.port() == Some(0) {
            return Err(HttpOriginError::PortZero);
        }

        let canonical = parsed.as_str().trim_end_matches('/');
        Ok(Self {
            canonical: Arc::from(canonical),
            host: Arc::from(host),
        })
    }

    /// Return the canonical origin string without a trailing slash.
    pub fn as_str(&self) -> &str {
        &self.canonical
    }

    /// Return the canonical host component.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Build one endpoint below this origin from an absolute path.
    pub fn endpoint(&self, absolute_path: &str) -> Result<Url, HttpOriginError> {
        if !absolute_path.starts_with('/')
            || absolute_path.contains('?')
            || absolute_path.contains('#')
        {
            return Err(HttpOriginError::InvalidEndpointPath(
                absolute_path.to_string(),
            ));
        }
        let mut endpoint = Url::parse(self.as_str())
            .expect("HttpOrigin retains the URL validated during construction");
        endpoint.set_path(absolute_path);
        Ok(endpoint)
    }
}

impl AsRef<str> for HttpOrigin {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for HttpOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for HttpOrigin {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for HttpOrigin {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

/// Invalid HTTP process origin or endpoint path.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum HttpOriginError {
    #[error("HTTP origin must use explicit http:// or https:// syntax")]
    InvalidSyntax,
    #[error("invalid HTTP origin: {0}")]
    InvalidUrl(String),
    #[error("HTTP origin must contain a host")]
    MissingHost,
    #[error("HTTP origin cannot contain credentials")]
    CredentialsNotAllowed,
    #[error("HTTP origin cannot contain a path")]
    PathNotAllowed,
    #[error("HTTP origin cannot contain a query")]
    QueryNotAllowed,
    #[error("HTTP origin cannot contain a fragment")]
    FragmentNotAllowed,
    #[error("HTTP origin port must be nonzero")]
    PortZero,
    #[error("HTTP endpoint path must be absolute and cannot contain a query or fragment: {0:?}")]
    InvalidEndpointPath(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_equivalent_http_origins() {
        let variants = [
            "HTTP://Example.COM:80",
            "http://example.com/",
            "http://example.com",
        ];
        let origins: Vec<HttpOrigin> = variants
            .into_iter()
            .map(|value| HttpOrigin::parse(value).unwrap())
            .collect();

        assert!(origins.iter().all(|origin| origin == &origins[0]));
        assert_eq!(origins[0].as_str(), "http://example.com");
        assert_eq!(origins[0].host(), "example.com");

        let https = HttpOrigin::parse("https://EXAMPLE.com:443/").unwrap();
        assert_eq!(https.as_str(), "https://example.com");
        assert_ne!(https, origins[0]);
    }

    #[test]
    fn preserves_semantically_distinct_origins() {
        assert_ne!(
            HttpOrigin::parse("http://example.com:8080").unwrap(),
            HttpOrigin::parse("http://example.com").unwrap()
        );
        assert_ne!(
            HttpOrigin::parse("http://127.0.0.1").unwrap(),
            HttpOrigin::parse("http://[::1]").unwrap()
        );
    }

    #[test]
    fn rejects_non_origins() {
        let invalid = [
            "example.com:30000",
            "grpc://example.com:30000",
            "ftp://example.com",
            "http://user@example.com",
            "http://@example.com",
            "http://example.com:",
            "http://example.com/path",
            "http://example.com?query=value",
            "http://example.com#fragment",
            "http:///missing-host",
            "http:example.com",
            "http:\\\\example.com",
            " http://example.com",
            "http://example.com/a/..",
            "http://example.com:0",
        ];

        for value in invalid {
            assert!(HttpOrigin::parse(value).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn builds_endpoints_without_string_concatenation() {
        let origin = HttpOrigin::parse("http://example.com:30000/").unwrap();
        assert_eq!(
            origin.endpoint("/server_info").unwrap().as_str(),
            "http://example.com:30000/server_info"
        );
        assert!(origin.endpoint("server_info").is_err());
        assert!(origin.endpoint("/server_info?raw=true").is_err());
    }

    #[test]
    fn serde_round_trip_revalidates_the_origin() {
        let origin = HttpOrigin::parse("HTTPS://EXAMPLE.com:443/").unwrap();
        let encoded = serde_json::to_string(&origin).unwrap();
        assert_eq!(encoded, r#""https://example.com""#);
        assert_eq!(
            serde_json::from_str::<HttpOrigin>(&encoded).unwrap(),
            origin
        );
        assert!(serde_json::from_str::<HttpOrigin>(r#""grpc://example.com""#).is_err());
    }
}
