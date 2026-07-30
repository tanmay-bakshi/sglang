use serde::{
    de::{Error as _, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use serde_json::{Map, Number, Value};
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub(crate) enum GenerateResponseExpectation<'a> {
    Scalar { request_id: &'a str },
    Batch { child_ids: &'a [String] },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAiResponseExpectation<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) choice_count: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ResponseValidationError {
    #[error("maximum JSON response size must be greater than zero")]
    InvalidResponseLimit,
    #[error("JSON response exceeds the configured {max_response_bytes}-byte limit")]
    ResponseTooLarge { max_response_bytes: usize },
    #[error("JSON response is malformed or ambiguous: {0}")]
    MalformedJson(String),
    #[error("response contains a structured error envelope")]
    StructuredError,
    #[error("response has the wrong top-level shape")]
    WrongTopLevelShape,
    #[error("response is missing required field {0}")]
    MissingField(&'static str),
    #[error("response field {0} has the wrong type")]
    WrongFieldType(&'static str),
    #[error("response ID does not match the expected request ID")]
    WrongRequestId,
    #[error("batch response child IDs do not match the expected ordered child IDs")]
    WrongChildIds,
    #[error("response choice count does not match the request")]
    WrongChoiceCount,
    #[error("response choice indices are not the canonical sequence")]
    InvalidChoiceIndices,
    #[error("response expectation is invalid")]
    InvalidExpectation,
}

pub(crate) fn validate_generate_response(
    body: &[u8],
    max_response_bytes: usize,
    expectation: GenerateResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    let value = parse_bounded_json(body, max_response_bytes)?;
    match expectation {
        GenerateResponseExpectation::Scalar { request_id } => {
            let object = value
                .as_object()
                .ok_or(ResponseValidationError::WrongTopLevelShape)?;
            validate_generate_item(object, request_id)
        }
        GenerateResponseExpectation::Batch { child_ids } => {
            validate_expected_child_ids(child_ids)?;
            let items = value
                .as_array()
                .ok_or(ResponseValidationError::WrongTopLevelShape)?;
            if items.len() != child_ids.len() {
                return Err(ResponseValidationError::WrongChildIds);
            }

            for (item, expected_id) in items.iter().zip(child_ids) {
                let object = item
                    .as_object()
                    .ok_or(ResponseValidationError::WrongTopLevelShape)?;
                let child_id = generate_item_id(object)?;
                validate_generate_item_shape(object)?;
                if child_id != expected_id {
                    return Err(ResponseValidationError::WrongChildIds);
                }
            }
            Ok(())
        }
    }
}

pub(crate) fn validate_chat_completion_response(
    body: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_openai_response(body, max_response_bytes, expectation, "chat.completion")
}

pub(crate) fn validate_generate_stream_event(
    payload: &[u8],
    max_response_bytes: usize,
    expectation: GenerateResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    let value = parse_bounded_json(payload, max_response_bytes)?;
    let object = value
        .as_object()
        .ok_or(ResponseValidationError::WrongTopLevelShape)?;
    let actual_id = generate_item_id(object)?;
    validate_generate_item_shape(object)?;

    match expectation {
        GenerateResponseExpectation::Scalar { request_id } => {
            let index = required_index(object, "index", "index")?;
            if index != 0 {
                return Err(ResponseValidationError::InvalidChoiceIndices);
            }
            if actual_id != request_id {
                return Err(ResponseValidationError::WrongRequestId);
            }
        }
        GenerateResponseExpectation::Batch { child_ids } => {
            validate_expected_child_ids(child_ids)?;
            let index = required_index(object, "index", "index")?;
            let expected_id = child_ids
                .get(index)
                .ok_or(ResponseValidationError::InvalidChoiceIndices)?;
            if actual_id != expected_id {
                return Err(ResponseValidationError::WrongChildIds);
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_chat_completion_stream_event(
    payload: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_openai_stream_event(
        payload,
        max_response_bytes,
        expectation,
        "chat.completion.chunk",
    )
}

pub(crate) fn validate_completion_stream_event(
    payload: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_openai_stream_event(payload, max_response_bytes, expectation, "text_completion")
}

pub(crate) fn validate_completion_response(
    body: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_openai_response(body, max_response_bytes, expectation, "text_completion")
}

fn validate_openai_stream_event(
    payload: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
    expected_object: &'static str,
) -> Result<(), ResponseValidationError> {
    if expectation.choice_count == 0 {
        return Err(ResponseValidationError::InvalidExpectation);
    }

    let value = parse_bounded_json(payload, max_response_bytes)?;
    let object = value
        .as_object()
        .ok_or(ResponseValidationError::WrongTopLevelShape)?;
    reject_error_envelope(object)?;
    if required_string(object, "id")? != expectation.request_id {
        return Err(ResponseValidationError::WrongRequestId);
    }
    if required_string(object, "object")? != expected_object {
        return Err(ResponseValidationError::WrongFieldType("object"));
    }
    required_u64(object, "created")?;
    required_string(object, "model")?;

    let choices = object
        .get("choices")
        .ok_or(ResponseValidationError::MissingField("choices"))?
        .as_array()
        .ok_or(ResponseValidationError::WrongFieldType("choices"))?;
    let mut previous_index = None;
    for choice in choices {
        let choice = choice
            .as_object()
            .ok_or(ResponseValidationError::WrongFieldType("choices[]"))?;
        let index = required_index(choice, "index", "choices[].index")?;
        if index >= expectation.choice_count
            || previous_index.is_some_and(|previous| index <= previous)
        {
            return Err(ResponseValidationError::InvalidChoiceIndices);
        }
        validate_openai_choice_shape(choice, expected_object, true)?;
        previous_index = Some(index);
    }
    Ok(())
}

fn validate_openai_response(
    body: &[u8],
    max_response_bytes: usize,
    expectation: OpenAiResponseExpectation<'_>,
    expected_object: &'static str,
) -> Result<(), ResponseValidationError> {
    if expectation.choice_count == 0 {
        return Err(ResponseValidationError::InvalidExpectation);
    }

    let value = parse_bounded_json(body, max_response_bytes)?;
    let object = value
        .as_object()
        .ok_or(ResponseValidationError::WrongTopLevelShape)?;
    reject_error_envelope(object)?;

    let request_id = required_string(object, "id")?;
    if request_id != expectation.request_id {
        return Err(ResponseValidationError::WrongRequestId);
    }
    if required_string(object, "object")? != expected_object {
        return Err(ResponseValidationError::WrongFieldType("object"));
    }
    required_u64(object, "created")?;
    required_string(object, "model")?;

    let choices = object
        .get("choices")
        .ok_or(ResponseValidationError::MissingField("choices"))?
        .as_array()
        .ok_or(ResponseValidationError::WrongFieldType("choices"))?;
    if choices.len() != expectation.choice_count {
        return Err(ResponseValidationError::WrongChoiceCount);
    }

    for (expected_index, choice) in choices.iter().enumerate() {
        let choice = choice
            .as_object()
            .ok_or(ResponseValidationError::WrongFieldType("choices[]"))?;
        let index = choice
            .get("index")
            .ok_or(ResponseValidationError::MissingField("choices[].index"))?
            .as_u64()
            .ok_or(ResponseValidationError::WrongFieldType("choices[].index"))?;
        if index != expected_index as u64 {
            return Err(ResponseValidationError::InvalidChoiceIndices);
        }
        validate_openai_choice_shape(choice, expected_object, false)?;
    }
    Ok(())
}

fn validate_expected_child_ids(child_ids: &[String]) -> Result<(), ResponseValidationError> {
    if child_ids.is_empty() {
        return Err(ResponseValidationError::InvalidExpectation);
    }
    let mut unique = std::collections::HashSet::with_capacity(child_ids.len());
    if !child_ids.iter().all(|child_id| unique.insert(child_id)) {
        return Err(ResponseValidationError::InvalidExpectation);
    }
    Ok(())
}

fn validate_generate_item(
    object: &Map<String, Value>,
    expected_id: &str,
) -> Result<(), ResponseValidationError> {
    let actual_id = generate_item_id(object)?;
    if actual_id != expected_id {
        return Err(ResponseValidationError::WrongRequestId);
    }
    validate_generate_item_shape(object)
}

fn generate_item_id(object: &Map<String, Value>) -> Result<&str, ResponseValidationError> {
    reject_error_envelope(object)?;
    let meta_info = object
        .get("meta_info")
        .ok_or(ResponseValidationError::MissingField("meta_info"))?
        .as_object()
        .ok_or(ResponseValidationError::WrongFieldType("meta_info"))?;
    required_string(meta_info, "id")
}

fn validate_generate_item_shape(
    object: &Map<String, Value>,
) -> Result<(), ResponseValidationError> {
    reject_error_envelope(object)?;
    required_string(object, "text")?;
    let output_ids = object
        .get("output_ids")
        .ok_or(ResponseValidationError::MissingField("output_ids"))?
        .as_array()
        .ok_or(ResponseValidationError::WrongFieldType("output_ids"))?;
    if output_ids.iter().any(|output_id| {
        output_id
            .as_u64()
            .is_none_or(|value| value > u32::MAX as u64)
    }) {
        return Err(ResponseValidationError::WrongFieldType("output_ids[]"));
    }
    object
        .get("meta_info")
        .ok_or(ResponseValidationError::MissingField("meta_info"))?
        .as_object()
        .ok_or(ResponseValidationError::WrongFieldType("meta_info"))?;
    Ok(())
}

fn reject_error_envelope(object: &Map<String, Value>) -> Result<(), ResponseValidationError> {
    if object.contains_key("error") {
        return Err(ResponseValidationError::StructuredError);
    }
    Ok(())
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, ResponseValidationError> {
    object
        .get(field)
        .ok_or(ResponseValidationError::MissingField(field))?
        .as_str()
        .ok_or(ResponseValidationError::WrongFieldType(field))
}

fn required_index(
    object: &Map<String, Value>,
    key: &'static str,
    field_name: &'static str,
) -> Result<usize, ResponseValidationError> {
    let index = object
        .get(key)
        .ok_or(ResponseValidationError::MissingField(field_name))?
        .as_u64()
        .ok_or(ResponseValidationError::WrongFieldType(field_name))?;
    usize::try_from(index).map_err(|_| ResponseValidationError::WrongFieldType(field_name))
}

fn required_u64(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<u64, ResponseValidationError> {
    object
        .get(field)
        .ok_or(ResponseValidationError::MissingField(field))?
        .as_u64()
        .ok_or(ResponseValidationError::WrongFieldType(field))
}

fn validate_openai_choice_shape(
    choice: &Map<String, Value>,
    object: &str,
    streaming: bool,
) -> Result<(), ResponseValidationError> {
    match (object, streaming) {
        ("chat.completion", false) => {
            choice
                .get("message")
                .ok_or(ResponseValidationError::MissingField("choices[].message"))?
                .as_object()
                .ok_or(ResponseValidationError::WrongFieldType("choices[].message"))?;
        }
        ("chat.completion.chunk", true) => {
            choice
                .get("delta")
                .ok_or(ResponseValidationError::MissingField("choices[].delta"))?
                .as_object()
                .ok_or(ResponseValidationError::WrongFieldType("choices[].delta"))?;
        }
        ("text_completion", _) => {
            required_string(choice, "text")?;
        }
        _ => return Err(ResponseValidationError::WrongFieldType("object")),
    }
    Ok(())
}

fn parse_bounded_json(
    body: &[u8],
    max_response_bytes: usize,
) -> Result<Value, ResponseValidationError> {
    if max_response_bytes == 0 {
        return Err(ResponseValidationError::InvalidResponseLimit);
    }
    if body.len() > max_response_bytes {
        return Err(ResponseValidationError::ResponseTooLarge { max_response_bytes });
    }

    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let value = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| ResponseValidationError::MalformedJson(error.to_string()))?
        .0;
    deserializer
        .end()
        .map_err(|error| ResponseValidationError::MalformedJson(error.to_string()))?;
    Ok(value)
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            let value = object.next_value::<UniqueValue>()?;
            values.insert(key, value.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_chat_completion_response, validate_chat_completion_stream_event,
        validate_completion_response, validate_completion_stream_event, validate_generate_response,
        validate_generate_stream_event, GenerateResponseExpectation, OpenAiResponseExpectation,
        ResponseValidationError,
    };

    const LIMIT: usize = 4_096;

    fn generate_item(id: &str) -> String {
        format!(
            r#"{{"text":"ok","output_ids":[1],"meta_info":{{"id":"{id}","finish_reason":{{"type":"stop"}}}}}}"#
        )
    }

    fn openai_expectation<'a>(
        request_id: &'a str,
        choice_count: usize,
    ) -> OpenAiResponseExpectation<'a> {
        OpenAiResponseExpectation {
            request_id,
            choice_count,
        }
    }

    #[test]
    fn validates_generate_scalar_and_batch_responses() {
        let scalar = generate_item("root");
        validate_generate_response(
            scalar.as_bytes(),
            LIMIT,
            GenerateResponseExpectation::Scalar { request_id: "root" },
        )
        .unwrap();

        let batch = format!(
            "[{},{}]",
            generate_item("child-a"),
            generate_item("child-b")
        );
        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        validate_generate_response(
            batch.as_bytes(),
            LIMIT,
            GenerateResponseExpectation::Batch {
                child_ids: &child_ids,
            },
        )
        .unwrap();
    }

    #[test]
    fn rejects_generate_shape_id_and_cardinality_failures() {
        let scalar = generate_item("wrong");
        assert_eq!(
            validate_generate_response(
                scalar.as_bytes(),
                LIMIT,
                GenerateResponseExpectation::Scalar { request_id: "root" },
            ),
            Err(ResponseValidationError::WrongRequestId)
        );

        let child_ids = vec!["a".to_owned(), "b".to_owned()];
        for body in [
            format!("[{}]", generate_item("a")),
            format!("[{},{}]", generate_item("a"), generate_item("a")),
            format!("[{},{}]", generate_item("a"), generate_item("c")),
            format!("[{},{}]", generate_item("b"), generate_item("a")),
        ] {
            assert_eq!(
                validate_generate_response(
                    body.as_bytes(),
                    LIMIT,
                    GenerateResponseExpectation::Batch {
                        child_ids: &child_ids,
                    },
                ),
                Err(ResponseValidationError::WrongChildIds)
            );
        }

        assert_eq!(
            validate_generate_response(
                b"[]",
                LIMIT,
                GenerateResponseExpectation::Scalar { request_id: "root" },
            ),
            Err(ResponseValidationError::WrongTopLevelShape)
        );
        assert_eq!(
            validate_generate_response(
                generate_item("a").as_bytes(),
                LIMIT,
                GenerateResponseExpectation::Batch {
                    child_ids: &child_ids,
                },
            ),
            Err(ResponseValidationError::WrongTopLevelShape)
        );
    }

    #[test]
    fn rejects_invalid_generate_expectations_and_shapes() {
        let empty = Vec::new();
        assert_eq!(
            validate_generate_response(
                b"[]",
                LIMIT,
                GenerateResponseExpectation::Batch { child_ids: &empty },
            ),
            Err(ResponseValidationError::InvalidExpectation)
        );

        let duplicates = vec!["a".to_owned(), "a".to_owned()];
        assert_eq!(
            validate_generate_response(
                b"[]",
                LIMIT,
                GenerateResponseExpectation::Batch {
                    child_ids: &duplicates,
                },
            ),
            Err(ResponseValidationError::InvalidExpectation)
        );

        for body in [
            r#"{"text":"ok","meta_info":{"id":"root"}}"#,
            r#"{"text":"ok","output_ids":[]}"#,
        ] {
            assert!(validate_generate_response(
                body.as_bytes(),
                LIMIT,
                GenerateResponseExpectation::Scalar { request_id: "root" },
            )
            .is_err());
        }

        let extended =
            r#"{"text":"ok","output_ids":[],"meta_info":{"id":"root"},"stage":"complete"}"#;
        validate_generate_response(
            extended.as_bytes(),
            LIMIT,
            GenerateResponseExpectation::Scalar { request_id: "root" },
        )
        .unwrap();
    }

    #[test]
    fn validates_chat_and_completion_responses() {
        let chat = br#"{
            "id":"chat-id",
            "object":"chat.completion",
            "created":1,
            "model":"model",
            "choices":[
                {"index":0,"message":{"role":"assistant","content":"a"},"finish_reason":"stop"},
                {"index":1,"message":{"role":"assistant","content":"b"},"finish_reason":"stop"}
            ],
            "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
        }"#;
        validate_chat_completion_response(chat, LIMIT, openai_expectation("chat-id", 2)).unwrap();

        let completion = br#"{
            "id":"completion-id",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[{"index":0,"text":"ok","finish_reason":"stop"}]
        }"#;
        validate_completion_response(completion, LIMIT, openai_expectation("completion-id", 1))
            .unwrap();
    }

    #[test]
    fn validates_route_specific_stream_events() {
        let scalar = r#"{"text":"ok","output_ids":[1],"meta_info":{"id":"root"},"index":0}"#;
        validate_generate_stream_event(
            scalar.as_bytes(),
            LIMIT,
            GenerateResponseExpectation::Scalar { request_id: "root" },
        )
        .unwrap();

        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        let batch =
            r#"{"text":"ok","output_ids":[1],"meta_info":{"id":"child-b"},"index":1}"#.to_owned();
        validate_generate_stream_event(
            batch.as_bytes(),
            LIMIT,
            GenerateResponseExpectation::Batch {
                child_ids: &child_ids,
            },
        )
        .unwrap();

        let chat = br#"{
            "id":"chat-id",
            "object":"chat.completion.chunk",
            "created":1,
            "model":"model",
            "choices":[{"index":1,"delta":{"content":"ok"}}]
        }"#;
        validate_chat_completion_stream_event(chat, LIMIT, openai_expectation("chat-id", 2))
            .unwrap();

        let completion = br#"{
            "id":"completion-id",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[],
            "usage":{}
        }"#;
        validate_completion_stream_event(completion, LIMIT, openai_expectation("completion-id", 2))
            .unwrap();
    }

    #[test]
    fn rejects_wrong_route_stream_events() {
        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        for body in [
            r#"{"text":"ok","output_ids":[],"meta_info":{"id":"root"}}"#,
            r#"{"text":"ok","output_ids":[],"meta_info":{"id":"root"},"index":1}"#,
        ] {
            assert!(validate_generate_stream_event(
                body.as_bytes(),
                LIMIT,
                GenerateResponseExpectation::Scalar { request_id: "root" },
            )
            .is_err());
        }

        for body in [
            r#"{"text":"ok","output_ids":[],"meta_info":{"id":"child-a"},"index":1}"#,
            r#"{"text":"ok","output_ids":[],"meta_info":{"id":"child-b"},"index":2}"#,
        ] {
            assert!(validate_generate_stream_event(
                body.as_bytes(),
                LIMIT,
                GenerateResponseExpectation::Batch {
                    child_ids: &child_ids,
                },
            )
            .is_err());
        }

        for body in [
            br#"{"id":"wrong","object":"chat.completion.chunk","choices":[{"index":0}]}"#
                .as_slice(),
            br#"{"id":"id","object":"chat.completion","choices":[{"index":0}]}"#,
            br#"{"id":"id","object":"chat.completion.chunk","choices":[{"index":2}]}"#,
            br#"{"id":"id","object":"chat.completion.chunk","choices":[{"index":1},{"index":0}]}"#,
            br#"{"error":{"message":"failed"}}"#,
        ] {
            assert!(validate_chat_completion_stream_event(
                body,
                LIMIT,
                openai_expectation("id", 2)
            )
            .is_err());
        }

        let chat_chunk = br#"{"id":"id","object":"chat.completion.chunk","choices":[{"index":0}]}"#;
        assert!(
            validate_completion_stream_event(chat_chunk, LIMIT, openai_expectation("id", 1))
                .is_err()
        );
    }

    #[test]
    fn rejects_wrong_openai_object_and_request_id() {
        let wrong_id = br#"{"id":"wrong","object":"chat.completion","choices":[{"index":0}]}"#;
        assert_eq!(
            validate_chat_completion_response(wrong_id, LIMIT, openai_expectation("expected", 1)),
            Err(ResponseValidationError::WrongRequestId)
        );

        let wrong_object =
            br#"{"id":"expected","object":"text_completion","choices":[{"index":0}]}"#;
        assert_eq!(
            validate_chat_completion_response(
                wrong_object,
                LIMIT,
                openai_expectation("expected", 1)
            ),
            Err(ResponseValidationError::WrongFieldType("object"))
        );
    }

    #[test]
    fn rejects_invalid_choice_counts_and_indices() {
        for body in [
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[]}"#
                .as_slice(),
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[{"index":0},{"index":1},{"index":2}]}"#,
        ] {
            assert_eq!(
                validate_chat_completion_response(body, LIMIT, openai_expectation("id", 2)),
                Err(ResponseValidationError::WrongChoiceCount)
            );
        }

        for body in [
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[{},{"index":1}]}"#.as_slice(),
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[{"index":0},{"index":0}]}"#,
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[{"index":1},{"index":0}]}"#,
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[{"index":0},{"index":2}]}"#,
        ] {
            assert!(
                validate_chat_completion_response(body, LIMIT, openai_expectation("id", 2))
                    .is_err()
            );
        }

        let body =
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[]}"#;
        assert_eq!(
            validate_chat_completion_response(body, LIMIT, openai_expectation("id", 0)),
            Err(ResponseValidationError::InvalidExpectation)
        );
    }

    #[test]
    fn rejects_error_envelopes_for_all_routes() {
        let error = br#"{"error":{"message":"failed","type":"server_error"}}"#;
        assert_eq!(
            validate_generate_response(
                error,
                LIMIT,
                GenerateResponseExpectation::Scalar { request_id: "id" }
            ),
            Err(ResponseValidationError::StructuredError)
        );
        assert_eq!(
            validate_chat_completion_response(error, LIMIT, openai_expectation("id", 1)),
            Err(ResponseValidationError::StructuredError)
        );
        assert_eq!(
            validate_completion_response(error, LIMIT, openai_expectation("id", 1)),
            Err(ResponseValidationError::StructuredError)
        );
    }

    #[test]
    fn rejects_malformed_oversized_trailing_and_ambiguous_json() {
        let expectation = GenerateResponseExpectation::Scalar { request_id: "id" };
        for body in [
            b"".as_slice(),
            b"{",
            b"null",
            b"{}{}",
            br#"{"text":"a","text":"b","output_ids":[],"meta_info":{"id":"id"}}"#,
            br#"{"text":"a","output_ids":[],"meta_info":{"id":"id","id":"other"}}"#,
        ] {
            assert!(validate_generate_response(body, LIMIT, expectation).is_err());
        }

        let valid = generate_item("id");
        assert_eq!(
            validate_generate_response(valid.as_bytes(), valid.len() - 1, expectation),
            Err(ResponseValidationError::ResponseTooLarge {
                max_response_bytes: valid.len() - 1
            })
        );
        assert_eq!(
            validate_generate_response(valid.as_bytes(), 0, expectation),
            Err(ResponseValidationError::InvalidResponseLimit)
        );
    }
}
