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
pub(crate) struct ChatResponseExpectation<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) choice_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompletionResponseExpectation<'a> {
    pub(crate) child_ids: &'a [String],
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
    expectation: ChatResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    if expectation.choice_count == 0 {
        return Err(ResponseValidationError::InvalidExpectation);
    }
    let value = parse_bounded_json(body, max_response_bytes)?;
    let (object, response_id) = validate_openai_identity(&value)?;
    if response_id != expectation.request_id {
        return Err(ResponseValidationError::WrongRequestId);
    }
    let choices = validate_openai_shape(object, "chat.completion")?;
    validate_canonical_choices(choices, expectation.choice_count, "chat.completion")
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
    expectation: ChatResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    if expectation.choice_count == 0 {
        return Err(ResponseValidationError::InvalidExpectation);
    }
    let value = parse_bounded_json(payload, max_response_bytes)?;
    let (object, response_id) = validate_openai_identity(&value)?;
    if response_id != expectation.request_id {
        return Err(ResponseValidationError::WrongRequestId);
    }
    let choices = validate_openai_shape(object, "chat.completion.chunk")?;
    validate_stream_choices(choices, expectation.choice_count, "chat.completion.chunk")?;
    Ok(())
}

pub(crate) fn validate_completion_stream_event(
    payload: &[u8],
    max_response_bytes: usize,
    expectation: CompletionResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_expected_child_ids(expectation.child_ids)?;
    let value = parse_bounded_json(payload, max_response_bytes)?;
    let (object, response_id) = validate_openai_identity(&value)?;
    let choices = validate_openai_shape(object, "text_completion")?;
    if choices.is_empty() {
        if !expectation
            .child_ids
            .iter()
            .any(|child_id| child_id == response_id)
        {
            return Err(ResponseValidationError::WrongChildIds);
        }
        return Ok(());
    }

    let mut previous_index = None;
    for choice in choices {
        let choice = choice
            .as_object()
            .ok_or(ResponseValidationError::WrongFieldType("choices[]"))?;
        let index = required_index(choice, "index", "choices[].index")?;
        if previous_index.is_some_and(|previous| index <= previous) {
            return Err(ResponseValidationError::InvalidChoiceIndices);
        }
        let expected_id = expectation
            .child_ids
            .get(index)
            .ok_or(ResponseValidationError::InvalidChoiceIndices)?;
        if response_id != expected_id {
            return Err(ResponseValidationError::WrongChildIds);
        }
        validate_openai_choice_shape(choice, "text_completion", true)?;
        previous_index = Some(index);
    }
    Ok(())
}

pub(crate) fn validate_completion_response(
    body: &[u8],
    max_response_bytes: usize,
    expectation: CompletionResponseExpectation<'_>,
) -> Result<(), ResponseValidationError> {
    validate_expected_child_ids(expectation.child_ids)?;
    let value = parse_bounded_json(body, max_response_bytes)?;
    let (object, response_id) = validate_openai_identity(&value)?;
    if response_id != expectation.child_ids[0] {
        return Err(ResponseValidationError::WrongChildIds);
    }
    let choices = validate_openai_shape(object, "text_completion")?;
    validate_canonical_choices(choices, expectation.child_ids.len(), "text_completion")
}

fn validate_openai_identity(
    value: &Value,
) -> Result<(&Map<String, Value>, &str), ResponseValidationError> {
    let object = value
        .as_object()
        .ok_or(ResponseValidationError::WrongTopLevelShape)?;
    reject_error_envelope(object)?;
    let response_id = required_string(object, "id")?;
    Ok((object, response_id))
}

fn validate_openai_shape<'a>(
    object: &'a Map<String, Value>,
    expected_object: &'static str,
) -> Result<&'a [Value], ResponseValidationError> {
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
    Ok(choices)
}

fn validate_stream_choices(
    choices: &[Value],
    choice_count: usize,
    expected_object: &'static str,
) -> Result<(), ResponseValidationError> {
    let mut previous_index = None;
    for choice in choices {
        let choice = choice
            .as_object()
            .ok_or(ResponseValidationError::WrongFieldType("choices[]"))?;
        let index = required_index(choice, "index", "choices[].index")?;
        if index >= choice_count || previous_index.is_some_and(|previous| index <= previous) {
            return Err(ResponseValidationError::InvalidChoiceIndices);
        }
        validate_openai_choice_shape(choice, expected_object, true)?;
        previous_index = Some(index);
    }
    Ok(())
}

fn validate_canonical_choices(
    choices: &[Value],
    choice_count: usize,
    expected_object: &'static str,
) -> Result<(), ResponseValidationError> {
    if choices.len() != choice_count {
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
    if child_ids.is_empty() || child_ids.iter().any(|child_id| child_id.is_empty()) {
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
        validate_generate_stream_event, ChatResponseExpectation, CompletionResponseExpectation,
        GenerateResponseExpectation, ResponseValidationError,
    };

    const LIMIT: usize = 4_096;

    fn generate_item(id: &str) -> String {
        format!(
            r#"{{"text":"ok","output_ids":[1],"meta_info":{{"id":"{id}","finish_reason":{{"type":"stop"}}}}}}"#
        )
    }

    fn chat_expectation<'a>(
        request_id: &'a str,
        choice_count: usize,
    ) -> ChatResponseExpectation<'a> {
        ChatResponseExpectation {
            request_id,
            choice_count,
        }
    }

    fn completion_expectation(child_ids: &[String]) -> CompletionResponseExpectation<'_> {
        CompletionResponseExpectation { child_ids }
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
        validate_chat_completion_response(chat, LIMIT, chat_expectation("chat-id", 2)).unwrap();

        let completion = br#"{
            "id":"completion-id",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[{"index":0,"text":"ok","finish_reason":"stop"}]
        }"#;
        let completion_ids = vec!["completion-id".to_owned()];
        validate_completion_response(completion, LIMIT, completion_expectation(&completion_ids))
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
        validate_chat_completion_stream_event(chat, LIMIT, chat_expectation("chat-id", 2)).unwrap();

        let completion_ids = vec!["completion-a".to_owned(), "completion-id".to_owned()];
        let completion = br#"{
            "id":"completion-id",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[],
            "usage":{}
        }"#;
        validate_completion_stream_event(
            completion,
            LIMIT,
            completion_expectation(&completion_ids),
        )
        .unwrap();
    }

    #[test]
    fn validates_ordered_completion_child_ids_for_non_stream_responses() {
        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        let response = br#"{
            "id":"child-a",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[
                {"index":0,"text":"first","finish_reason":"stop"},
                {"index":1,"text":"second","finish_reason":"stop"}
            ]
        }"#;
        validate_completion_response(response, LIMIT, completion_expectation(&child_ids)).unwrap();

        let wrong_response_id = br#"{
            "id":"child-b",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[
                {"index":0,"text":"first"},
                {"index":1,"text":"second"}
            ]
        }"#;
        assert_eq!(
            validate_completion_response(
                wrong_response_id,
                LIMIT,
                completion_expectation(&child_ids),
            ),
            Err(ResponseValidationError::WrongChildIds)
        );

        let wrong_choice_count = br#"{
            "id":"child-a",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[{"index":0,"text":"first"}]
        }"#;
        assert_eq!(
            validate_completion_response(
                wrong_choice_count,
                LIMIT,
                completion_expectation(&child_ids),
            ),
            Err(ResponseValidationError::WrongChoiceCount)
        );

        let noncanonical_indices = br#"{
            "id":"child-a",
            "object":"text_completion",
            "created":1,
            "model":"model",
            "choices":[
                {"index":0,"text":"first"},
                {"index":2,"text":"second"}
            ]
        }"#;
        assert_eq!(
            validate_completion_response(
                noncanonical_indices,
                LIMIT,
                completion_expectation(&child_ids),
            ),
            Err(ResponseValidationError::InvalidChoiceIndices)
        );
    }

    #[test]
    fn validates_per_child_completion_stream_ids_and_response_level_events() {
        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        for event in [
            br#"{"id":"child-a","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"a"}]}"#.as_slice(),
            br#"{"id":"child-b","object":"text_completion","created":1,"model":"model","choices":[{"index":1,"text":"b"}]}"#,
            br#"{"id":"child-a","object":"text_completion","created":1,"model":"model","choices":[],"usage":{"prompt_tokens":2}}"#,
            br#"{"id":"child-b","object":"text_completion","created":1,"model":"model","choices":[],"usage":{"prompt_tokens":2}}"#,
        ] {
            validate_completion_stream_event(
                event,
                LIMIT,
                completion_expectation(&child_ids),
            )
            .unwrap();
        }
    }

    #[test]
    fn rejects_completion_stream_ids_that_do_not_match_choice_indices() {
        let child_ids = vec!["child-a".to_owned(), "child-b".to_owned()];
        for event in [
            br#"{"id":"child-a","object":"text_completion","created":1,"model":"model","choices":[{"index":1,"text":"wrong child"}]}"#.as_slice(),
            br#"{"id":"child-a","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"a"},{"index":1,"text":"b"}]}"#,
            br#"{"id":"unknown","object":"text_completion","created":1,"model":"model","choices":[]}"#,
            br#"{"id":"child-b","object":"text_completion","created":1,"model":"model","choices":[{"index":2,"text":"out of range"}]}"#,
        ] {
            assert!(validate_completion_stream_event(
                event,
                LIMIT,
                completion_expectation(&child_ids),
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_empty_or_duplicate_completion_expectations() {
        for child_ids in [
            Vec::new(),
            vec![String::new()],
            vec!["child".to_owned(), "child".to_owned()],
        ] {
            assert_eq!(
                validate_completion_response(b"{}", LIMIT, completion_expectation(&child_ids),),
                Err(ResponseValidationError::InvalidExpectation)
            );
            assert_eq!(
                validate_completion_stream_event(b"{}", LIMIT, completion_expectation(&child_ids),),
                Err(ResponseValidationError::InvalidExpectation)
            );
        }
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
            assert!(
                validate_chat_completion_stream_event(body, LIMIT, chat_expectation("id", 2))
                    .is_err()
            );
        }

        let chat_chunk = br#"{"id":"id","object":"chat.completion.chunk","choices":[{"index":0}]}"#;
        let completion_ids = vec!["id".to_owned()];
        assert!(validate_completion_stream_event(
            chat_chunk,
            LIMIT,
            completion_expectation(&completion_ids),
        )
        .is_err());
    }

    #[test]
    fn rejects_wrong_openai_object_and_request_id() {
        let wrong_id = br#"{"id":"wrong","object":"chat.completion","choices":[{"index":0}]}"#;
        assert_eq!(
            validate_chat_completion_response(wrong_id, LIMIT, chat_expectation("expected", 1)),
            Err(ResponseValidationError::WrongRequestId)
        );

        let wrong_object =
            br#"{"id":"expected","object":"text_completion","choices":[{"index":0}]}"#;
        assert_eq!(
            validate_chat_completion_response(wrong_object, LIMIT, chat_expectation("expected", 1)),
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
                validate_chat_completion_response(body, LIMIT, chat_expectation("id", 2)),
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
                validate_chat_completion_response(body, LIMIT, chat_expectation("id", 2))
                    .is_err()
            );
        }

        let body =
            br#"{"id":"id","object":"chat.completion","created":1,"model":"model","choices":[]}"#;
        assert_eq!(
            validate_chat_completion_response(body, LIMIT, chat_expectation("id", 0)),
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
            validate_chat_completion_response(error, LIMIT, chat_expectation("id", 1)),
            Err(ResponseValidationError::StructuredError)
        );
        let completion_ids = vec!["id".to_owned()];
        assert_eq!(
            validate_completion_response(error, LIMIT, completion_expectation(&completion_ids)),
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
