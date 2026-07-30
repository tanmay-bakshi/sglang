use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SseProgress {
    InProgress {
        events: Vec<Vec<u8>>,
    },
    Terminal {
        events: Vec<Vec<u8>>,
        consumed_bytes: usize,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum SseValidationError {
    #[error("maximum SSE event size must be greater than zero")]
    InvalidEventLimit,
    #[error("SSE event exceeds the configured {max_event_bytes}-byte limit")]
    EventTooLarge { max_event_bytes: usize },
    #[error("SSE stream contains a malformed line ending")]
    MalformedLineEnding,
    #[error("SSE event is empty")]
    EmptyEvent,
    #[error("SSE event contains an unsupported field")]
    UnsupportedField,
    #[error("SSE event contains more than one data field")]
    MultipleDataFields,
    #[error("SSE terminal marker contains invalid whitespace")]
    InvalidTerminalMarker,
    #[error("SSE stream contains bytes after the terminal event")]
    TrailingData,
    #[error("SSE stream ended before a complete terminal event")]
    Truncated,
}

pub(crate) struct SseParser {
    max_event_bytes: usize,
    event_bytes: usize,
    line: Vec<u8>,
    data_seen: bool,
    terminal_event: bool,
    event_payload: Vec<u8>,
    pending_cr: bool,
    terminal: bool,
}

impl SseParser {
    pub(crate) fn new(max_event_bytes: usize) -> Result<Self, SseValidationError> {
        if max_event_bytes == 0 {
            return Err(SseValidationError::InvalidEventLimit);
        }

        Ok(Self {
            max_event_bytes,
            event_bytes: 0,
            line: Vec::new(),
            data_seen: false,
            terminal_event: false,
            event_payload: Vec::new(),
            pending_cr: false,
            terminal: false,
        })
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<SseProgress, SseValidationError> {
        if self.terminal {
            if chunk.is_empty() {
                return Ok(SseProgress::Terminal {
                    events: Vec::new(),
                    consumed_bytes: 0,
                });
            }
            return Err(SseValidationError::TrailingData);
        }

        let mut events = Vec::new();
        for (offset, &byte) in chunk.iter().enumerate() {
            self.count_event_byte()?;
            if self.pending_cr {
                if byte != b'\n' {
                    return Err(SseValidationError::MalformedLineEnding);
                }
                self.pending_cr = false;
                self.complete_line(&mut events)?;
                if self.terminal {
                    return Ok(SseProgress::Terminal {
                        events,
                        consumed_bytes: offset + 1,
                    });
                }
                continue;
            }

            match byte {
                b'\r' => self.pending_cr = true,
                b'\n' => self.complete_line(&mut events)?,
                _ => self.line.push(byte),
            }
            if self.terminal {
                return Ok(SseProgress::Terminal {
                    events,
                    consumed_bytes: offset + 1,
                });
            }
        }

        Ok(SseProgress::InProgress { events })
    }

    pub(crate) fn finish(self) -> Result<(), SseValidationError> {
        if self.terminal {
            return Ok(());
        }
        if self.pending_cr {
            return Err(SseValidationError::MalformedLineEnding);
        }
        Err(SseValidationError::Truncated)
    }

    fn count_event_byte(&mut self) -> Result<(), SseValidationError> {
        self.event_bytes += 1;
        if self.event_bytes <= self.max_event_bytes {
            return Ok(());
        }
        Err(SseValidationError::EventTooLarge {
            max_event_bytes: self.max_event_bytes,
        })
    }

    fn complete_line(&mut self, events: &mut Vec<Vec<u8>>) -> Result<(), SseValidationError> {
        if self.line.is_empty() {
            if !self.data_seen {
                return Err(SseValidationError::EmptyEvent);
            }

            if self.terminal_event {
                self.terminal = true;
                return Ok(());
            }

            events.push(std::mem::take(&mut self.event_payload));
            self.data_seen = false;
            self.event_bytes = 0;
            return Ok(());
        }

        if !self.line.starts_with(b"data:") {
            return Err(SseValidationError::UnsupportedField);
        }
        if self.data_seen {
            return Err(SseValidationError::MultipleDataFields);
        }

        let mut payload = &self.line[b"data:".len()..];
        if payload.first() == Some(&b' ') {
            payload = &payload[1..];
        }

        let trimmed_payload = trim_ascii(payload);
        if trimmed_payload == b"[DONE]" && payload != b"[DONE]" {
            return Err(SseValidationError::InvalidTerminalMarker);
        }

        self.data_seen = true;
        self.terminal_event = payload == b"[DONE]";
        if !self.terminal_event {
            self.event_payload.extend_from_slice(payload);
        }
        self.line.clear();
        Ok(())
    }
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{SseParser, SseProgress, SseValidationError};

    const LIMIT: usize = 1_024;

    fn validate_chunks(stream: &[u8], split_positions: &[usize]) -> Result<(), SseValidationError> {
        let mut validator = SseParser::new(LIMIT)?;
        let mut start = 0;
        for &end in split_positions {
            validator.push(&stream[start..end])?;
            start = end;
        }
        validator.push(&stream[start..])?;
        validator.finish()
    }

    fn assert_every_two_way_split_is_valid(stream: &[u8]) {
        for split in 0..=stream.len() {
            validate_chunks(stream, &[split])
                .unwrap_or_else(|error| panic!("split {split} failed: {error}"));
        }
    }

    #[test]
    fn accepts_every_split_for_lf_and_crlf_streams() {
        assert_every_two_way_split_is_valid(b"data: {\"text\":\"ok\"}\n\ndata: [DONE]\n\n");
        assert_every_two_way_split_is_valid(b"data: {\"text\":\"ok\"}\r\n\r\ndata: [DONE]\r\n\r\n");
    }

    #[test]
    fn accepts_terminal_split_at_every_byte_boundary() {
        let prefix = b"data: payload\n\n";
        let terminal = b"data: [DONE]\n\n";
        for split in 0..=terminal.len() {
            let mut validator = SseParser::new(LIMIT).unwrap();
            assert_eq!(
                validator.push(prefix).unwrap(),
                SseProgress::InProgress {
                    events: vec![b"payload".to_vec()]
                }
            );
            let first_terminal_part = validator.push(&terminal[..split]).unwrap();
            if split == terminal.len() {
                assert_eq!(
                    first_terminal_part,
                    SseProgress::Terminal {
                        events: Vec::new(),
                        consumed_bytes: terminal.len(),
                    }
                );
            } else {
                assert_eq!(
                    first_terminal_part,
                    SseProgress::InProgress { events: Vec::new() }
                );
            }
            assert_eq!(
                validator.push(&terminal[split..]).unwrap(),
                SseProgress::Terminal {
                    events: Vec::new(),
                    consumed_bytes: terminal.len() - split,
                }
            );
            validator.finish().unwrap();
        }
    }

    #[test]
    fn accepts_multiple_nonterminal_events() {
        validate_chunks(
            b"data:first\n\ndata: second\n\ndata:{\"value\":\"[DONE]\"}\n\ndata:[DONE]\n\n",
            &[1, 7, 13, 19, 31, 46],
        )
        .unwrap();
    }

    #[test]
    fn rejects_unsupported_fields_and_multiple_data_lines() {
        for stream in [
            b": comment\n\ndata: [DONE]\n\n".as_slice(),
            b"event: message\n\ndata: [DONE]\n\n",
            b"id: 1\n\ndata: [DONE]\n\n",
            b"retry: 1\n\ndata: [DONE]\n\n",
        ] {
            assert_eq!(
                validate_chunks(stream, &[]),
                Err(SseValidationError::UnsupportedField)
            );
        }

        assert_eq!(
            validate_chunks(b"data: one\ndata: two\n\n", &[]),
            Err(SseValidationError::MultipleDataFields)
        );
    }

    #[test]
    fn rejects_empty_events_and_malformed_line_endings() {
        assert_eq!(
            validate_chunks(b"\ndata: [DONE]\n\n", &[]),
            Err(SseValidationError::EmptyEvent)
        );
        assert_eq!(
            validate_chunks(b"data: one\n\n\ndata: [DONE]\n\n", &[]),
            Err(SseValidationError::EmptyEvent)
        );
        assert_eq!(
            validate_chunks(b"data: one\rdata: [DONE]\n\n", &[]),
            Err(SseValidationError::MalformedLineEnding)
        );
        assert_eq!(
            validate_chunks(b"data: one\r", &[]),
            Err(SseValidationError::MalformedLineEnding)
        );
    }

    #[test]
    fn rejects_incomplete_streams() {
        for stream in [
            b"".as_slice(),
            b"data: payload",
            b"data: payload\n",
            b"data: payload\n\n",
            b"data: [DONE]",
            b"data: [DONE]\n",
        ] {
            assert_eq!(
                validate_chunks(stream, &[]),
                Err(SseValidationError::Truncated),
                "unexpected result for {stream:?}"
            );
        }
    }

    #[test]
    fn rejects_whitespace_modified_terminal_markers() {
        for payload in [
            b" [DONE] ".as_slice(),
            b"  [DONE]",
            b"\t[DONE]",
            b"[DONE] ",
            b"[DONE]\t",
        ] {
            let mut stream = b"data:".to_vec();
            stream.extend_from_slice(payload);
            stream.extend_from_slice(b"\n\n");
            assert_eq!(
                validate_chunks(&stream, &[]),
                Err(SseValidationError::InvalidTerminalMarker),
                "unexpected result for {payload:?}"
            );
        }
    }

    #[test]
    fn does_not_misclassify_done_inside_payloads() {
        for payload in [
            b"{\"text\":\"[DONE]\"}".as_slice(),
            b"prefix[DONE]",
            b"[DONE]suffix",
            b"\"[DONE]\"",
        ] {
            let mut stream = b"data: ".to_vec();
            stream.extend_from_slice(payload);
            stream.extend_from_slice(b"\n\ndata: [DONE]\n\n");
            validate_chunks(&stream, &[]).unwrap();
        }
    }

    #[test]
    fn reports_the_terminal_boundary_and_ignores_same_chunk_suffix() {
        let terminal = b"data: [DONE]\n\n";
        for byte in 0..=u8::MAX {
            let mut stream = terminal.to_vec();
            stream.push(byte);
            assert_eq!(
                SseParser::new(LIMIT).unwrap().push(&stream),
                Ok(SseProgress::Terminal {
                    events: Vec::new(),
                    consumed_bytes: terminal.len(),
                }),
                "wrong boundary before trailing byte {byte}"
            );
        }

        let mut validator = SseParser::new(LIMIT).unwrap();
        assert_eq!(
            validator.push(terminal).unwrap(),
            SseProgress::Terminal {
                events: Vec::new(),
                consumed_bytes: terminal.len(),
            }
        );
        assert_eq!(validator.push(b"x"), Err(SseValidationError::TrailingData));
    }

    #[test]
    fn enforces_event_limit_including_line_endings() {
        let stream = b"data:long-payload\n\ndata:[DONE]\n\n";
        let first_event_bytes = b"data:long-payload\n\n".len();

        let mut exact = SseParser::new(first_event_bytes).unwrap();
        exact.push(stream).unwrap();
        exact.finish().unwrap();

        let mut too_small = SseParser::new(first_event_bytes - 1).unwrap();
        assert_eq!(
            too_small.push(stream),
            Err(SseValidationError::EventTooLarge {
                max_event_bytes: first_event_bytes - 1
            })
        );
        assert!(matches!(
            SseParser::new(0),
            Err(SseValidationError::InvalidEventLimit)
        ));
    }

    #[test]
    fn emits_every_completed_nonterminal_payload_once() {
        let mut validator = SseParser::new(LIMIT).unwrap();
        assert_eq!(
            validator.push(b"data: one\n\ndata: tw").unwrap(),
            SseProgress::InProgress {
                events: vec![b"one".to_vec()]
            }
        );
        assert_eq!(
            validator
                .push(b"o\n\ndata: three\n\ndata: [DONE]\n\n")
                .unwrap(),
            SseProgress::Terminal {
                events: vec![b"two".to_vec(), b"three".to_vec()],
                consumed_bytes: b"o\n\ndata: three\n\ndata: [DONE]\n\n".len(),
            }
        );
        validator.finish().unwrap();
    }
}
