#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <sys/random.h>
#include <sys/timerfd.h>
#include <system_error>
#include <thread>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

#include "native_producer_api.h"

namespace py = pybind11;

namespace {

constexpr std::size_t kGenerationBytes = 16;
constexpr std::size_t kDigestBytes = 32;
constexpr std::size_t kReceiptNonceBytes = 16;
constexpr std::uint64_t kSourceReclaimableMask = (1ULL << 8) - 1;
constexpr std::uint64_t kSourceResourceMask = (1ULL << 9) - 1;
constexpr std::uint64_t kDecodeResourceMask = ((1ULL << 16) - 1) ^ kSourceResourceMask;
constexpr std::size_t kQualificationMeasuredHopCount = 7;
constexpr std::size_t kQualificationLifecycleHopCount = 10;
constexpr std::size_t kQualificationMaximumMachineCount = 64;
constexpr std::uint64_t kQualificationMaximumFullPathSampleCount = 8'000'000;

using Generation = std::array<std::uint8_t, kGenerationBytes>;
using Digest = std::array<std::uint8_t, kDigestBytes>;
using Nonce = std::array<std::uint8_t, kReceiptNonceBytes>;

enum class OwnerRole : std::uint8_t { kSource = 1, kDecode = 2 };

enum class EventKind : std::uint16_t {
  kSourceSubmissionAccepted = 10,
  kSourceProducerCompleted = 11,
  kSourceGatherPosted = 12,
  kSourceNativeTerminal = 13,
  kSourceOutcomesSent = 14,
  kSourceTeardownReceived = 15,
  kSourceAckSent = 16,
  kSourceRequestReady = 17,
  kSourceReclaimConsumed = 18,
  kSourceGatewayPublished = 19,
  kSourcePublicationFailed = 20,
  kSourceRequestFailed = 21,
  kSourceOwnerDied = 22,
  kSourcePublisherDied = 23,
  kSourceInboxOverflow = 24,
  kDecodeAllocationPublished = 40,
  kDecodeWriterAggregationStarted = 41,
  kDecodeWriterManifestCompleted = 42,
  kDecodeScatterStarted = 43,
  kDecodeScatterTerminal = 44,
  kDecodeTeardownSent = 45,
  kDecodeAckAggregationStarted = 46,
  kDecodeAckManifestCompleted = 47,
  kDecodeAdoptionConsumed = 48,
  kDecodeMetadataConsumed = 49,
  kDecodeLocalReadyIssued = 50,
  kDecodeRequestReady = 51,
  kDecodeCancelUnpublished = 52,
  kDecodeRequestFailed = 53,
  kDecodeOwnerDied = 54,
  kDecodeInboxOverflow = 55,
};

enum class ReceiptKind : std::uint8_t {
  kNone = 0,
  kAdoptionReady = 1,
  kMetadataConsumed = 2,
  kLocalDecodeReady = 3,
  kRequestReady = 4,
  kReclaimAuthorized = 5,
  kReclaimConsumed = 6,
  kGatewayPublished = 7,
  kFailure = 8,
};

enum class ReceiptOutcome : std::uint8_t {
  kNone = 0,
  kSuccess = 1,
  kFailure = 2,
  kCancelled = 3,
};

enum class SourcePhase : std::uint8_t {
  kFrozen = 1,
  kWaitingForProducer = 2,
  kGathering = 3,
  kNativeInFlight = 4,
  kLocalTransferTerminal = 5,
  kOutcomesSent = 6,
  kTeardownReceived = 7,
  kAckSent = 8,
  kRequestReadyReceived = 9,
  kPublicationQuarantined = 10,
  kRetired = 11,
  kQuarantined = 12,
};

enum class DecodePhase : std::uint8_t {
  kPrepared = 1,
  kPublished = 2,
  kWriterAggregating = 3,
  kScatterReady = 4,
  kScatterInFlight = 5,
  kScatterTerminal = 6,
  kTeardownSent = 7,
  kAckAggregating = 8,
  kAdoptionReady = 9,
  kAdoptedByScheduler = 10,
  kMetadataConsumed = 11,
  kLocalDecodeReady = 12,
  kRequestReady = 13,
  kRetired = 14,
  kQuarantined = 15,
};

enum class ActionKind : std::uint8_t {
  kReclaimAuthorized = 1,
  kAdoptionReady = 2,
  kLocalDecodeReady = 3,
  kRequestRetired = 4,
  kRequestQuarantined = 5,
  kProcessFatal = 6,
  kSourceGatherReady = 7,
  kSourceOutcomeReady = 8,
  kSourceAckReady = 9,
  kDecodeScatterReady = 10,
  kDecodeTeardownReady = 11,
  kGatewayPublicationReady = 12,
};

enum class DeadlineKind : std::uint8_t {
  kExistingNixlCapabilityReady = 1,
  kExistingPackedControl = 2,
  kOwnerProducerAndGather = 3,
  kOwnerNativeTransfer = 4,
  kOwnerDecodeScatter = 5,
  kOwnerTeardownAck = 6,
  kOwnerRequestGlobalReady = 7,
  kOwnerSchedulerReceiptConsumption = 8,
  kOwnerGatewayPublication = 9,
  kOwnerShutdownDrain = 10,
};

enum class FatalCode : std::uint8_t {
  kNone = 0,
  kInputQueueOverflow = 1,
  kOutputQueueOverflow = 2,
  kEventfdFailure = 3,
  kTimerfdFailure = 4,
  kProducerSequence = 5,
  kDuplicateBinding = 6,
  kUnknownBinding = 7,
  kIllegalTransition = 8,
  kReceiptAuthority = 9,
  kReceiptReplay = 10,
  kUnknownAction = 11,
  kActionReplay = 12,
  kDeadlineExpiry = 13,
  kDependencyDeath = 14,
  kCloseWithRetainedInventory = 15,
  kInternalError = 16,
  kPendingCallQueueFailure = 17,
  kHandoffTimeout = 18,
  kHandoffAuthority = 19,
};

enum class ProducerClass : std::uint8_t {
  kLocal = 1,
  kReceipt = 2,
  kControl = 3,
  kQualification = 4,
};

struct ProcessIdentity {
  Generation process_generation{};
  OwnerRole role{OwnerRole::kSource};
  std::uint32_t tp_rank{0};
  std::uint32_t tp_size{0};
  Digest digest{};
};

struct RequestBinding {
  std::uint64_t room_id{0};
  Generation request_generation{};
  ProcessIdentity owner{};
  Digest rank_manifest_digest{};
  Digest allocation_digest{};
  Digest digest{};
};

struct PublicationIdentity {
  std::uint64_t room_id{0};
  Generation request_generation{};
  Generation publisher_process_generation{};
  Generation publication_generation{};
  Digest digest{};
};

struct Receipt {
  RequestBinding binding{};
  ProcessIdentity issuer{};
  ReceiptKind kind{ReceiptKind::kNone};
  ReceiptOutcome outcome{ReceiptOutcome::kNone};
  std::uint64_t terminal_timestamp_ns{0};
  Nonce nonce{};
};

struct Event {
  std::uint64_t producer_id{0};
  std::uint64_t producer_sequence{0};
  Digest binding_digest{};
  EventKind kind{EventKind::kSourceSubmissionAccepted};
  std::uint64_t enqueued_ns{0};
  bool has_receipt{false};
  Receipt receipt{};
  std::int32_t reason_code{0};
  std::int64_t backend_status{0};
  std::string reason{};
};

struct DeadlineSpec {
  DeadlineKind kind{DeadlineKind::kOwnerProducerAndGather};
  std::uint64_t duration_ns{0};
  bool process_fatal{false};
  std::string starts_at{};
  std::string timeout_outcome{};
};

struct Action {
  std::uint64_t action_id{0};
  ActionKind kind{ActionKind::kRequestRetired};
  RequestBinding binding{};
  std::uint64_t commit_timestamp_ns{0};
  std::optional<Receipt> receipt{};
};

struct Output {
  RequestBinding binding{};
  std::uint64_t owner_sequence{0};
  std::uint64_t producer_id{0};
  std::uint64_t producer_sequence{0};
  std::uint64_t enqueued_ns{0};
  std::uint64_t completed_ns{0};
  EventKind event_kind{EventKind::kSourceSubmissionAccepted};
  OwnerRole role{OwnerRole::kSource};
  std::uint8_t previous_phase{0};
  std::uint8_t phase{0};
  std::uint64_t live_resources{0};
  std::uint64_t retired_resources{0};
  std::uint64_t quarantined_resources{0};
  std::uint16_t armed_deadline_mask{0};
  bool process_fatal{false};
  FatalCode fatal_code{FatalCode::kNone};
  std::vector<Action> actions{};
};

struct Observation {
  RequestBinding binding{};
  std::uint64_t owner_sequence{0};
  std::uint64_t producer_id{0};
  std::uint64_t producer_sequence{0};
  std::uint32_t producer_rank{0};
  EventKind event_kind{EventKind::kSourceSubmissionAccepted};
  std::uint64_t enqueued_ns{0};
  std::uint64_t completed_ns{0};
  OwnerRole role{OwnerRole::kSource};
};

struct Trace {
  Event event{};
  std::uint64_t completed_ns{0};
  std::uint64_t owner_sequence{0};
  std::uint8_t previous_phase{0};
  std::uint8_t current_phase{0};
  std::size_t machine_index{0};
  std::uint64_t generation_index{0};
  std::uint8_t hop_index{0};
};

struct LatencyStatistics {
  std::uint64_t count{0};
  std::uint64_t p50_ns{0};
  std::uint64_t p95_ns{0};
  std::uint64_t p99_ns{0};
  std::uint64_t maximum_ns{0};
};

struct DigestHash {
  std::size_t operator()(const Digest &value) const noexcept {
    std::size_t result = 1469598103934665603ULL;
    for (const std::uint8_t byte : value) {
      result ^= static_cast<std::size_t>(byte);
      result *= 1099511628211ULL;
    }
    return result;
  }
};

struct DigestEqual {
  bool operator()(const Digest &left, const Digest &right) const noexcept {
    return left == right;
  }
};

struct NonceHash {
  std::size_t operator()(const Nonce &value) const noexcept {
    std::size_t result = 1469598103934665603ULL;
    for (const std::uint8_t byte : value) {
      result ^= static_cast<std::size_t>(byte);
      result *= 1099511628211ULL;
    }
    return result;
  }
};

struct ProducerRegistration {
  std::uint64_t producer_id{0};
  std::string name{};
  ProducerClass producer_class{ProducerClass::kLocal};
  OwnerRole allowed_role{OwnerRole::kSource};
  bool has_issuer{false};
  ProcessIdentity issuer{};
  std::uint64_t next_submission_sequence{0};
  std::uint64_t next_dispatch_sequence{0};
  bool retirement_requested{false};
  bool retired{false};
};

struct Lifecycle {
  RequestBinding binding{};
  std::optional<PublicationIdentity> publication_identity{};
  std::unordered_set<Digest, DigestHash, DigestEqual> trusted_issuers{};
  std::unordered_set<Nonce, NonceHash> consumed_receipts{};
  OwnerRole role{OwnerRole::kSource};
  std::uint8_t phase{0};
  std::uint64_t live_resources{0};
  std::uint64_t retired_resources{0};
  std::uint64_t quarantined_resources{0};
  std::uint16_t armed_deadline_mask{0};
  std::array<std::uint64_t, 11> deadline_expiry_ns{};
  bool reclaim_authorized{false};
  bool reclaim_consumed{false};
  bool publication_authorized{false};
  bool gateway_published{false};
  bool publication_quarantined{false};
};

enum class InputKind : std::uint8_t {
  kRegisterLifecycle = 1,
  kEvent = 2,
  kRetireProducer = 3,
};

struct InputCommand {
  InputKind kind{InputKind::kEvent};
  Lifecycle lifecycle{};
  Event event{};
  std::uint64_t producer_id{0};
  std::uint64_t retire_after_sequence{0};
};

struct QualificationState {
  bool running{false};
  bool draining{false};
  bool complete{false};
  std::size_t machine_count{0};
  std::uint64_t minimum_duration_ns{0};
  std::uint64_t minimum_transition_count{0};
  std::uint64_t started_ns{0};
  std::uint64_t ended_ns{0};
  std::uint64_t transition_count{0};
  std::uint64_t lifecycle_transition_count{0};
  std::uint64_t sample_count{0};
  std::uint64_t owner_sequence_start{0};
  std::uint64_t owner_sequence_end{0};
  bool summary_complete{false};
  std::vector<Digest> bindings{};
  std::vector<std::uint64_t> producer_ids{};
  std::vector<std::uint64_t> producer_sequences{};
  std::vector<std::uint64_t> generations{};
  std::vector<std::uint64_t> completed_generations{};
  std::vector<std::uint8_t> hops{};
  std::vector<std::uint64_t> path_latency_accumulators{};
  std::array<std::vector<std::uint64_t>, kQualificationMeasuredHopCount>
      hop_latencies{};
  std::vector<std::uint64_t> path_latencies{};
  std::array<LatencyStatistics, kQualificationMeasuredHopCount>
      hop_statistics{};
  LatencyStatistics path_statistics{};
  std::vector<std::array<std::optional<Trace>,
                         kQualificationMeasuredHopCount>>
      first_audit_traces{};
  std::vector<std::array<std::optional<Trace>,
                         kQualificationMeasuredHopCount>>
      last_audit_traces{};
};

std::uint64_t monotonic_raw_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "CLOCK_MONOTONIC_RAW failed");
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

template <std::size_t Size>
std::array<std::uint8_t, Size> exact_bytes(const py::handle &value,
                                          const char *name) {
  const std::string bytes = py::cast<py::bytes>(value);
  if (bytes.size() != Size) {
    throw std::invalid_argument(std::string(name) + " has the wrong width");
  }
  std::array<std::uint8_t, Size> result{};
  std::memcpy(result.data(), bytes.data(), Size);
  return result;
}

template <std::size_t Size>
py::bytes to_bytes(const std::array<std::uint8_t, Size> &value) {
  return py::bytes(reinterpret_cast<const char *>(value.data()), Size);
}

bool same_process_identity(const ProcessIdentity &left,
                           const ProcessIdentity &right) noexcept {
  return left.process_generation == right.process_generation &&
         left.role == right.role && left.tp_rank == right.tp_rank &&
         left.tp_size == right.tp_size && left.digest == right.digest;
}

bool same_binding(const RequestBinding &left,
                  const RequestBinding &right) noexcept {
  return left.room_id == right.room_id &&
         left.request_generation == right.request_generation &&
         same_process_identity(left.owner, right.owner) &&
         left.rank_manifest_digest == right.rank_manifest_digest &&
         left.allocation_digest == right.allocation_digest &&
         left.digest == right.digest;
}

ProcessIdentity process_identity_from_python(const py::dict &value) {
  ProcessIdentity result{};
  result.process_generation =
      exact_bytes<kGenerationBytes>(value["process_generation"],
                                    "process_generation");
  result.role = static_cast<OwnerRole>(py::cast<int>(value["role"]));
  result.tp_rank = py::cast<std::uint32_t>(value["tp_rank"]);
  result.tp_size = py::cast<std::uint32_t>(value["tp_size"]);
  result.digest = exact_bytes<kDigestBytes>(value["digest"], "process digest");
  if (result.role != OwnerRole::kSource && result.role != OwnerRole::kDecode) {
    throw std::invalid_argument("process role is invalid");
  }
  if (result.tp_size == 0 || result.tp_rank >= result.tp_size) {
    throw std::invalid_argument("process TP identity is invalid");
  }
  return result;
}

RequestBinding binding_from_python(const py::dict &value) {
  RequestBinding result{};
  result.room_id = py::cast<std::uint64_t>(value["room_id"]);
  result.request_generation =
      exact_bytes<kGenerationBytes>(value["request_generation"],
                                    "request_generation");
  result.owner = process_identity_from_python(py::cast<py::dict>(value["owner"]));
  result.rank_manifest_digest = exact_bytes<kDigestBytes>(
      value["rank_manifest_digest"], "rank_manifest_digest");
  result.allocation_digest = exact_bytes<kDigestBytes>(
      value["allocation_digest"], "allocation_digest");
  result.digest = exact_bytes<kDigestBytes>(value["digest"], "binding digest");
  return result;
}

PublicationIdentity publication_from_python(const py::dict &value) {
  PublicationIdentity result{};
  result.room_id = py::cast<std::uint64_t>(value["room_id"]);
  result.request_generation =
      exact_bytes<kGenerationBytes>(value["request_generation"],
                                    "request_generation");
  result.publisher_process_generation = exact_bytes<kGenerationBytes>(
      value["publisher_process_generation"], "publisher_process_generation");
  result.publication_generation = exact_bytes<kGenerationBytes>(
      value["publication_generation"], "publication_generation");
  result.digest =
      exact_bytes<kDigestBytes>(value["digest"], "publication digest");
  return result;
}

Receipt receipt_from_python(const py::dict &value) {
  Receipt result{};
  result.binding = binding_from_python(py::cast<py::dict>(value["binding"]));
  result.issuer = process_identity_from_python(py::cast<py::dict>(value["issuer"]));
  result.kind = static_cast<ReceiptKind>(py::cast<int>(value["kind"]));
  result.outcome = static_cast<ReceiptOutcome>(py::cast<int>(value["outcome"]));
  result.terminal_timestamp_ns =
      py::cast<std::uint64_t>(value["terminal_timestamp_ns"]);
  result.nonce = exact_bytes<kReceiptNonceBytes>(value["nonce"], "receipt nonce");
  return result;
}

Event event_from_python(const py::dict &value) {
  Event result{};
  result.producer_id = py::cast<std::uint64_t>(value["producer_id"]);
  result.binding_digest =
      exact_bytes<kDigestBytes>(value["binding_digest"], "binding digest");
  result.kind = static_cast<EventKind>(py::cast<int>(value["kind"]));
  result.enqueued_ns = py::cast<std::uint64_t>(value["enqueued_ns"]);
  if (value.contains("reason_code")) {
    result.reason_code = py::cast<std::int32_t>(value["reason_code"]);
  }
  if (value.contains("backend_status")) {
    result.backend_status = py::cast<std::int64_t>(value["backend_status"]);
  }
  const py::handle receipt = value["receipt"];
  if (!receipt.is_none()) {
    result.has_receipt = true;
    result.receipt = receipt_from_python(py::cast<py::dict>(receipt));
  }
  const py::handle reason = value["reason"];
  if (!reason.is_none()) {
    result.reason = py::cast<std::string>(reason);
  }
  return result;
}

py::dict process_identity_to_python(const ProcessIdentity &value) {
  py::dict result;
  result["process_generation"] = to_bytes(value.process_generation);
  result["role"] = static_cast<int>(value.role);
  result["tp_rank"] = value.tp_rank;
  result["tp_size"] = value.tp_size;
  result["digest"] = to_bytes(value.digest);
  return result;
}

py::dict binding_to_python(const RequestBinding &value) {
  py::dict result;
  result["room_id"] = value.room_id;
  result["request_generation"] = to_bytes(value.request_generation);
  result["owner"] = process_identity_to_python(value.owner);
  result["rank_manifest_digest"] = to_bytes(value.rank_manifest_digest);
  result["allocation_digest"] = to_bytes(value.allocation_digest);
  result["digest"] = to_bytes(value.digest);
  return result;
}

py::dict receipt_to_python(const Receipt &value) {
  py::dict result;
  result["binding"] = binding_to_python(value.binding);
  result["issuer"] = process_identity_to_python(value.issuer);
  result["kind"] = static_cast<int>(value.kind);
  result["outcome"] = static_cast<int>(value.outcome);
  result["terminal_timestamp_ns"] = value.terminal_timestamp_ns;
  result["nonce"] = to_bytes(value.nonce);
  return result;
}

py::dict action_to_python(const Action &value) {
  py::dict result;
  result["action_id"] = value.action_id;
  result["kind"] = static_cast<int>(value.kind);
  result["binding"] = binding_to_python(value.binding);
  result["commit_timestamp_ns"] = value.commit_timestamp_ns;
  if (value.receipt.has_value()) {
    result["receipt"] = receipt_to_python(value.receipt.value());
  } else {
    result["receipt"] = py::none();
  }
  return result;
}

py::dict output_to_python(const Output &value) {
  py::dict result;
  result["binding"] = binding_to_python(value.binding);
  result["owner_sequence"] = value.owner_sequence;
  result["producer_id"] = value.producer_id;
  result["producer_sequence"] = value.producer_sequence;
  result["enqueued_ns"] = value.enqueued_ns;
  result["completed_ns"] = value.completed_ns;
  result["event_kind"] = static_cast<int>(value.event_kind);
  result["role"] = static_cast<int>(value.role);
  result["previous_phase"] = value.previous_phase;
  result["phase"] = value.phase;
  result["live_resources"] = value.live_resources;
  result["retired_resources"] = value.retired_resources;
  result["quarantined_resources"] = value.quarantined_resources;
  result["armed_deadline_mask"] = value.armed_deadline_mask;
  result["process_fatal"] = value.process_fatal;
  result["fatal_code"] = static_cast<int>(value.fatal_code);
  py::list actions;
  for (const Action &action : value.actions) {
    actions.append(action_to_python(action));
  }
  result["actions"] = std::move(actions);
  return result;
}

py::dict observation_to_python(const Observation &value) {
  py::dict result;
  result["binding"] = binding_to_python(value.binding);
  result["owner_sequence"] = value.owner_sequence;
  result["producer_id"] = value.producer_id;
  result["producer_sequence"] = value.producer_sequence;
  result["producer_rank"] = value.producer_rank;
  result["event_kind"] = static_cast<int>(value.event_kind);
  result["enqueued_ns"] = value.enqueued_ns;
  result["completed_ns"] = value.completed_ns;
  result["role"] = static_cast<int>(value.role);
  return result;
}

py::dict trace_to_python(const Trace &value) {
  py::dict result;
  result["machine_index"] = value.machine_index;
  result["generation_index"] = value.generation_index;
  result["hop_index"] = value.hop_index;
  result["binding_digest"] = to_bytes(value.event.binding_digest);
  result["event_kind"] = static_cast<int>(value.event.kind);
  result["completed_ns"] = value.completed_ns;
  result["previous_phase"] = value.previous_phase;
  result["phase"] = value.current_phase;
  result["enqueued_ns"] = value.event.enqueued_ns;
  return result;
}

py::dict latency_statistics_to_python(const LatencyStatistics &value) {
  py::dict result;
  result["count"] = value.count;
  result["p50_ns"] = value.p50_ns;
  result["p95_ns"] = value.p95_ns;
  result["p99_ns"] = value.p99_ns;
  result["maximum_ns"] = value.maximum_ns;
  return result;
}

const char *fatal_name(FatalCode code) noexcept {
  switch (code) {
  case FatalCode::kNone: return "none";
  case FatalCode::kInputQueueOverflow: return "input_queue_overflow";
  case FatalCode::kOutputQueueOverflow: return "output_queue_overflow";
  case FatalCode::kEventfdFailure: return "eventfd_failure";
  case FatalCode::kTimerfdFailure: return "timerfd_failure";
  case FatalCode::kProducerSequence: return "producer_sequence";
  case FatalCode::kDuplicateBinding: return "duplicate_binding";
  case FatalCode::kUnknownBinding: return "unknown_binding";
  case FatalCode::kIllegalTransition: return "illegal_transition";
  case FatalCode::kReceiptAuthority: return "receipt_authority";
  case FatalCode::kReceiptReplay: return "receipt_replay";
  case FatalCode::kUnknownAction: return "unknown_action";
  case FatalCode::kActionReplay: return "action_replay";
  case FatalCode::kDeadlineExpiry: return "deadline_expiry";
  case FatalCode::kDependencyDeath: return "dependency_death";
  case FatalCode::kCloseWithRetainedInventory: return "close_with_retained_inventory";
  case FatalCode::kInternalError: return "internal_error";
  case FatalCode::kPendingCallQueueFailure: return "pending_call_queue_failure";
  case FatalCode::kHandoffTimeout: return "handoff_timeout";
  case FatalCode::kHandoffAuthority: return "handoff_authority";
  }
  return "unknown";
}

class NativeTerminalOwnerBridge;

struct SharedOwner : std::enable_shared_from_this<SharedOwner> {
  std::mutex mutex{};
  std::condition_variable condition{};
  std::condition_variable qualification_condition{};
  std::size_t input_capacity{0};
  std::size_t output_capacity{0};
  std::size_t observation_capacity{0};
  std::size_t maximum_live_lifecycles{0};
  ProcessIdentity owner_identity{};
  Digest deadline_table_digest{};
  std::array<DeadlineSpec, 11> deadline_specs{};
  std::unordered_map<std::uint64_t, ProducerRegistration> producers{};
  std::unordered_map<Digest, Lifecycle, DigestHash, DigestEqual> lifecycles{};
  std::deque<InputCommand> input_queue{};
  std::deque<Output> output_queue{};
  std::deque<Output> fatal_output_queue{};
  std::deque<Observation> observation_queue{};
  std::unordered_map<std::uint64_t, Action> pending_actions{};
  std::unordered_set<std::uint64_t> consumed_actions{};
  std::unordered_set<std::uint64_t> output_drain_action_ids{};
  std::unordered_set<std::uint64_t> handoff_action_ids{};
  std::unordered_set<Nonce, NonceHash> minted_nonces{};
  QualificationState qualification{};
  std::thread reactor{};
  int input_fd{-1};
  int output_fd{-1};
  int observation_fd{-1};
  int timer_fd{-1};
  int shutdown_fd{-1};
  std::uint64_t next_owner_sequence{0};
  std::uint64_t next_action_id{1};
  std::uint64_t total_action_count{0};
  std::uint64_t observation_count{0};
  std::uint64_t delivered_observation_count{0};
  std::uint64_t dropped_observation_count{0};
  std::uint64_t observation_eventfd_error_count{0};
  std::uint64_t next_handoff_callback_id{1};
  std::uint64_t scheduled_handoff_callback_id{0};
  std::uint64_t scheduled_handoff_watermark{0};
  std::uint64_t active_handoff_callback_id{0};
  std::uint64_t active_handoff_watermark{0};
  std::uint64_t handoff_callback_count{0};
  std::uint64_t handoff_completion_count{0};
  bool started{false};
  bool admission_open{true};
  bool event_admission_open{true};
  bool stop_requested{false};
  bool producer_join_requested{false};
  bool producer_join_barrier_complete{false};
  bool producers_joined{false};
  bool abort_started{false};
  bool reactor_stopped{false};
  bool closed{false};
  bool output_drain_active{false};
  bool observation_wake_armed{false};
  bool handoff_enabled{false};
  bool handoff_callback_scheduled{false};
  bool handoff_callback_active{false};
#ifdef SGLANG_TERMINAL_OWNER_TESTING
  bool test_clock_enabled{false};
  std::uint64_t test_now_ns{0};
#endif
  FatalCode fatal_code{FatalCode::kNone};
  int fatal_system_error{0};
  Digest fatal_binding{};
  std::uint64_t fatal_producer_id{0};
  std::uint64_t fatal_producer_sequence{0};
  std::string fatal_reason{};
  std::int32_t fatal_reason_code{0};
  std::int64_t fatal_backend_status{0};
};

struct PendingHandoffCall {
  std::shared_ptr<SharedOwner> owner{};
  std::uint64_t callback_id{0};
};

int terminal_handoff_pending_call(void *opaque) noexcept;

void register_forward_independent_handoffs_locked(
    SharedOwner &owner, const Output &output) noexcept;

void fail_forward_independent_handoff_locked(
    SharedOwner &owner, std::uint64_t action_id) noexcept;

std::uint64_t owner_now_ns_locked(const SharedOwner &owner) {
#ifdef SGLANG_TERMINAL_OWNER_TESTING
  if (owner.test_clock_enabled) {
    return owner.test_now_ns;
  }
#else
  static_cast<void>(owner);
#endif
  return monotonic_raw_ns();
}

struct ProducerCapsule {
  std::shared_ptr<SharedOwner> owner{};
  std::uint64_t producer_id{0};
};

void close_fd(int &fd) noexcept {
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

void signal_fd_locked(SharedOwner &owner, int fd) noexcept {
  if (fd < 0) {
    if (owner.fatal_code == FatalCode::kNone) {
      owner.fatal_code = FatalCode::kEventfdFailure;
      owner.fatal_system_error = EBADF;
    }
    return;
  }
  constexpr std::uint64_t increment = 1;
  for (;;) {
    const ssize_t result = ::write(fd, &increment, sizeof(increment));
    if (result == static_cast<ssize_t>(sizeof(increment))) {
      return;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (owner.fatal_code == FatalCode::kNone) {
      owner.fatal_code = FatalCode::kEventfdFailure;
      owner.fatal_system_error = result < 0 ? errno : EIO;
    }
    return;
  }
}

bool signal_observation_fd_locked(SharedOwner &owner) noexcept {
  if (owner.observation_wake_armed) {
    return true;
  }
  if (owner.observation_fd < 0) {
    ++owner.observation_eventfd_error_count;
    return false;
  }
  constexpr std::uint64_t increment = 1;
  for (;;) {
    const ssize_t result =
        ::write(owner.observation_fd, &increment, sizeof(increment));
    if (result == static_cast<ssize_t>(sizeof(increment))) {
      owner.observation_wake_armed = true;
      return true;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    ++owner.observation_eventfd_error_count;
    return false;
  }
}

int consume_fd(int fd) noexcept {
  std::uint64_t observed = 0;
  for (;;) {
    const ssize_t result = ::read(fd, &observed, sizeof(observed));
    if (result == static_cast<ssize_t>(sizeof(observed))) {
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0 && errno == EAGAIN) {
      return 0;
    }
    return result < 0 ? errno : EIO;
  }
}

int consume_timer_fd(int fd) noexcept {
  std::uint64_t expirations = 0;
  for (;;) {
    const ssize_t result = ::read(fd, &expirations, sizeof(expirations));
    if (result == static_cast<ssize_t>(sizeof(expirations))) {
      return 0;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    return result < 0 ? errno : EIO;
  }
}

Nonce mint_nonce_locked(SharedOwner &owner, const RequestBinding &binding,
                        ReceiptKind kind, std::uint64_t timestamp_ns) {
  static_cast<void>(binding);
  static_cast<void>(kind);
  static_cast<void>(timestamp_ns);
  for (;;) {
    Nonce nonce{};
    std::size_t offset = 0;
    while (offset < nonce.size()) {
      const ssize_t result =
          ::getrandom(nonce.data() + offset, nonce.size() - offset, 0);
      if (result > 0) {
        offset += static_cast<std::size_t>(result);
        continue;
      }
      if (result < 0 && errno == EINTR) {
        continue;
      }
      throw std::system_error(result < 0 ? errno : EIO,
                              std::generic_category(),
                              "receipt nonce getrandom failed");
    }
    if (owner.minted_nonces.insert(nonce).second) {
      return nonce;
    }
  }
}

Receipt mint_receipt_locked(SharedOwner &owner, const RequestBinding &binding,
                            ReceiptKind kind, ReceiptOutcome outcome,
                            std::uint64_t timestamp_ns) {
  Receipt receipt{};
  receipt.binding = binding;
  receipt.issuer = owner.owner_identity;
  receipt.kind = kind;
  receipt.outcome = outcome;
  receipt.terminal_timestamp_ns = timestamp_ns;
  receipt.nonce = mint_nonce_locked(owner, binding, kind, timestamp_ns);
  return receipt;
}

std::uint16_t deadline_bit(DeadlineKind kind) {
  return static_cast<std::uint16_t>(1U << (static_cast<unsigned>(kind) - 1U));
}

void arm_deadline_locked(SharedOwner &owner, Lifecycle &lifecycle,
                         DeadlineKind kind, std::uint64_t started_ns) {
  const std::uint8_t index = static_cast<std::uint8_t>(kind);
  if (index == 0 || index >= owner.deadline_specs.size()) {
    throw std::runtime_error("deadline kind is outside the frozen table");
  }
  const DeadlineSpec &spec = owner.deadline_specs[index];
  if (spec.duration_ns == 0) {
    throw std::runtime_error("deadline table is incomplete");
  }
  const std::uint16_t bit = deadline_bit(kind);
  if ((lifecycle.armed_deadline_mask & bit) != 0) {
    throw std::runtime_error("deadline is already armed");
  }
  lifecycle.armed_deadline_mask |= bit;
  lifecycle.deadline_expiry_ns[index] = started_ns + spec.duration_ns;
}

void cancel_deadline_locked(Lifecycle &lifecycle, DeadlineKind kind) noexcept {
  const std::uint8_t index = static_cast<std::uint8_t>(kind);
  lifecycle.armed_deadline_mask &= ~deadline_bit(kind);
  lifecycle.deadline_expiry_ns[index] = 0;
}

bool lifecycle_terminal(const Lifecycle &lifecycle) noexcept {
  if (lifecycle.role == OwnerRole::kSource) {
    const SourcePhase phase = static_cast<SourcePhase>(lifecycle.phase);
    return phase == SourcePhase::kRetired || phase == SourcePhase::kQuarantined;
  }
  const DecodePhase phase = static_cast<DecodePhase>(lifecycle.phase);
  return phase == DecodePhase::kRetired || phase == DecodePhase::kQuarantined;
}

void quarantine_live(Lifecycle &lifecycle) noexcept {
  lifecycle.quarantined_resources |= lifecycle.live_resources;
  lifecycle.live_resources = 0;
  lifecycle.armed_deadline_mask = 0;
  lifecycle.deadline_expiry_ns.fill(0);
  lifecycle.phase = lifecycle.role == OwnerRole::kSource
                        ? static_cast<std::uint8_t>(SourcePhase::kQuarantined)
                        : static_cast<std::uint8_t>(DecodePhase::kQuarantined);
}

void verify_conservation(const Lifecycle &lifecycle) {
  const std::uint64_t universe = lifecycle.role == OwnerRole::kSource
                                     ? kSourceResourceMask
                                     : kDecodeResourceMask;
  if ((lifecycle.live_resources & lifecycle.retired_resources) != 0 ||
      (lifecycle.live_resources & lifecycle.quarantined_resources) != 0 ||
      (lifecycle.retired_resources & lifecycle.quarantined_resources) != 0 ||
      (lifecycle.live_resources | lifecycle.retired_resources |
       lifecycle.quarantined_resources) != universe) {
    throw std::runtime_error("resource partition violated conservation");
  }
}

bool event_requires_receipt(EventKind kind) noexcept {
  switch (kind) {
  case EventKind::kSourceRequestReady:
  case EventKind::kSourceReclaimConsumed:
  case EventKind::kSourceGatewayPublished:
  case EventKind::kSourcePublicationFailed:
  case EventKind::kSourceRequestFailed:
  case EventKind::kDecodeAdoptionConsumed:
  case EventKind::kDecodeRequestReady:
  case EventKind::kDecodeRequestFailed:
    return true;
  default:
    return false;
  }
}

bool event_is_control_ingress(EventKind kind) noexcept {
  switch (kind) {
  case EventKind::kSourceTeardownReceived:
  case EventKind::kDecodeWriterAggregationStarted:
  case EventKind::kDecodeWriterManifestCompleted:
  case EventKind::kDecodeAckAggregationStarted:
  case EventKind::kDecodeAckManifestCompleted:
    return true;
  default:
    return false;
  }
}

bool producer_authorizes_event(const ProducerRegistration &producer,
                               const Event &event) noexcept {
  if (producer.producer_class == ProducerClass::kQualification) {
    return true;
  }
  const bool local_failure =
      producer.producer_class == ProducerClass::kLocal &&
      (event.kind == EventKind::kSourceRequestFailed ||
       event.kind == EventKind::kDecodeRequestFailed) &&
      !event.has_receipt;
  if (local_failure) {
    return true;
  }
  if (event_requires_receipt(event.kind)) {
    return producer.producer_class == ProducerClass::kReceipt;
  }
  if (event_is_control_ingress(event.kind)) {
    return producer.producer_class == ProducerClass::kControl;
  }
  return producer.producer_class == ProducerClass::kLocal;
}

std::pair<ReceiptKind, ReceiptOutcome> expected_receipt(EventKind kind) {
  switch (kind) {
  case EventKind::kSourceRequestReady:
  case EventKind::kDecodeRequestReady:
    return {ReceiptKind::kRequestReady, ReceiptOutcome::kSuccess};
  case EventKind::kSourceReclaimConsumed:
    return {ReceiptKind::kReclaimConsumed, ReceiptOutcome::kSuccess};
  case EventKind::kSourceGatewayPublished:
    return {ReceiptKind::kGatewayPublished, ReceiptOutcome::kSuccess};
  case EventKind::kDecodeAdoptionConsumed:
    return {ReceiptKind::kAdoptionReady, ReceiptOutcome::kSuccess};
  case EventKind::kSourcePublicationFailed:
  case EventKind::kSourceRequestFailed:
  case EventKind::kDecodeRequestFailed:
    return {ReceiptKind::kFailure, ReceiptOutcome::kFailure};
  default:
    return {ReceiptKind::kNone, ReceiptOutcome::kNone};
  }
}

void validate_receipt_locked(Lifecycle &lifecycle,
                             const ProducerRegistration &producer,
                             const Event &event) {
  const bool local_failure =
      producer.producer_class == ProducerClass::kLocal &&
      (event.kind == EventKind::kSourceRequestFailed ||
       event.kind == EventKind::kDecodeRequestFailed);
  if (local_failure) {
    if (event.has_receipt) {
      throw std::runtime_error("local failure producer cannot forge a receipt");
    }
    return;
  }
  if (event_is_control_ingress(event.kind)) {
    if (event.has_receipt) {
      throw std::runtime_error("control ingress supplied an unexpected receipt");
    }
    if (!producer.has_issuer ||
        lifecycle.trusted_issuers.count(producer.issuer.digest) != 1) {
      throw std::runtime_error(
          "control route issuer is not trusted by the lifecycle");
    }
    return;
  }
  if (!event_requires_receipt(event.kind)) {
    if (event.has_receipt) {
      throw std::runtime_error("event supplied an unexpected receipt");
    }
    return;
  }
  if (!event.has_receipt || !producer.has_issuer) {
    throw std::runtime_error("receipt-bearing event lacks issuer authority");
  }
  const Receipt &receipt = event.receipt;
  if (!same_binding(receipt.binding, lifecycle.binding) ||
      receipt.binding.digest != event.binding_digest) {
    throw std::runtime_error("receipt targets another exact binding");
  }
  if (!same_process_identity(receipt.issuer, producer.issuer) ||
      lifecycle.trusted_issuers.count(receipt.issuer.digest) != 1) {
    throw std::runtime_error("route, receipt, and lifecycle issuer disagree");
  }
  const auto expected = expected_receipt(event.kind);
  if (receipt.kind != expected.first || receipt.outcome != expected.second) {
    throw std::runtime_error("receipt authority does not match the event");
  }
  if (!lifecycle.consumed_receipts.insert(receipt.nonce).second) {
    throw std::runtime_error("receipt authority was replayed");
  }
}

void add_action_locked(SharedOwner &owner, Lifecycle &lifecycle, Output &output,
                       ActionKind kind, std::uint64_t completed_ns,
                       std::optional<ReceiptKind> receipt_kind = std::nullopt) {
  Action action{};
  action.action_id = owner.next_action_id++;
  action.kind = kind;
  action.binding = lifecycle.binding;
  action.commit_timestamp_ns = completed_ns;
  if (receipt_kind.has_value()) {
    action.receipt = mint_receipt_locked(owner, lifecycle.binding,
                                         receipt_kind.value(),
                                         ReceiptOutcome::kSuccess,
                                         completed_ns);
  }
  owner.pending_actions.emplace(action.action_id, action);
  ++owner.total_action_count;
  output.actions.push_back(std::move(action));
}

void discard_unpublished_actions_locked(SharedOwner &owner,
                                        const Output &output) noexcept {
  for (const Action &action : output.actions) {
    owner.pending_actions.erase(action.action_id);
  }
}

bool reduce_source_locked(SharedOwner &owner, Lifecycle &lifecycle,
                          const Event &event, Output &output,
                          std::uint64_t completed_ns) {
  const SourcePhase phase = static_cast<SourcePhase>(lifecycle.phase);
  if (phase == SourcePhase::kRetired || phase == SourcePhase::kQuarantined) {
    throw std::runtime_error("source lifecycle is terminal");
  }
  auto transition = [&](SourcePhase expected, SourcePhase next) {
    if (phase != expected) {
      throw std::runtime_error("source event is illegal from current phase");
    }
    lifecycle.phase = static_cast<std::uint8_t>(next);
  };
  switch (event.kind) {
  case EventKind::kSourceSubmissionAccepted:
    transition(SourcePhase::kFrozen, SourcePhase::kWaitingForProducer);
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerProducerAndGather, completed_ns);
    break;
  case EventKind::kSourceProducerCompleted:
    transition(SourcePhase::kWaitingForProducer, SourcePhase::kGathering);
    add_action_locked(owner, lifecycle, output, ActionKind::kSourceGatherReady,
                      completed_ns);
    break;
  case EventKind::kSourceGatherPosted:
    transition(SourcePhase::kGathering, SourcePhase::kNativeInFlight);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerProducerAndGather);
    arm_deadline_locked(owner, lifecycle, DeadlineKind::kOwnerNativeTransfer,
                        completed_ns);
    break;
  case EventKind::kSourceNativeTerminal:
    transition(SourcePhase::kNativeInFlight,
               SourcePhase::kLocalTransferTerminal);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerNativeTransfer);
    add_action_locked(owner, lifecycle, output, ActionKind::kSourceOutcomeReady,
                      completed_ns);
    break;
  case EventKind::kSourceOutcomesSent:
    transition(SourcePhase::kLocalTransferTerminal, SourcePhase::kOutcomesSent);
    arm_deadline_locked(owner, lifecycle, DeadlineKind::kOwnerTeardownAck,
                        completed_ns);
    break;
  case EventKind::kSourceTeardownReceived:
    transition(SourcePhase::kOutcomesSent, SourcePhase::kTeardownReceived);
    add_action_locked(owner, lifecycle, output, ActionKind::kSourceAckReady,
                      completed_ns);
    break;
  case EventKind::kSourceAckSent:
    transition(SourcePhase::kTeardownReceived, SourcePhase::kAckSent);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerTeardownAck);
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerRequestGlobalReady, completed_ns);
    break;
  case EventKind::kSourceRequestReady:
    transition(SourcePhase::kAckSent, SourcePhase::kRequestReadyReceived);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerRequestGlobalReady);
    lifecycle.reclaim_authorized = true;
    lifecycle.publication_authorized = true;
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerSchedulerReceiptConsumption,
                        completed_ns);
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerGatewayPublication, completed_ns);
    add_action_locked(owner, lifecycle, output, ActionKind::kReclaimAuthorized,
                      completed_ns, ReceiptKind::kReclaimAuthorized);
    add_action_locked(owner, lifecycle, output,
                      ActionKind::kGatewayPublicationReady, completed_ns);
    break;
  case EventKind::kSourceReclaimConsumed:
    if ((phase != SourcePhase::kRequestReadyReceived &&
         phase != SourcePhase::kPublicationQuarantined) ||
        lifecycle.reclaim_consumed) {
      throw std::runtime_error("source reclaim consumption is illegal");
    }
    lifecycle.live_resources &= ~kSourceReclaimableMask;
    lifecycle.retired_resources |= kSourceReclaimableMask;
    lifecycle.reclaim_consumed = true;
    cancel_deadline_locked(lifecycle,
                           DeadlineKind::kOwnerSchedulerReceiptConsumption);
    if (lifecycle.gateway_published) {
      lifecycle.phase = static_cast<std::uint8_t>(SourcePhase::kRetired);
      add_action_locked(owner, lifecycle, output, ActionKind::kRequestRetired,
                        completed_ns);
    }
    break;
  case EventKind::kSourceGatewayPublished:
    if (phase != SourcePhase::kRequestReadyReceived ||
        lifecycle.gateway_published || lifecycle.publication_quarantined) {
      throw std::runtime_error("source gateway publication is illegal");
    }
    lifecycle.live_resources &= ~(1ULL << 8);
    lifecycle.retired_resources |= 1ULL << 8;
    lifecycle.gateway_published = true;
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerGatewayPublication);
    if (lifecycle.reclaim_consumed) {
      lifecycle.phase = static_cast<std::uint8_t>(SourcePhase::kRetired);
      add_action_locked(owner, lifecycle, output, ActionKind::kRequestRetired,
                        completed_ns);
    }
    break;
  case EventKind::kSourcePublicationFailed:
  case EventKind::kSourcePublisherDied:
    if (phase != SourcePhase::kRequestReadyReceived ||
        !lifecycle.publication_authorized || lifecycle.gateway_published ||
        lifecycle.publication_quarantined) {
      throw std::runtime_error("publication failure lacks global authority");
    }
    lifecycle.live_resources &= ~(1ULL << 8);
    lifecycle.quarantined_resources |= 1ULL << 8;
    lifecycle.publication_quarantined = true;
    lifecycle.phase =
        static_cast<std::uint8_t>(SourcePhase::kPublicationQuarantined);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerGatewayPublication);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestQuarantined,
                      completed_ns);
    break;
  case EventKind::kSourceRequestFailed:
    if (lifecycle.live_resources == 0) {
      throw std::runtime_error(
          "source failure cannot quarantine exhausted resources");
    }
    quarantine_live(lifecycle);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestQuarantined,
                      completed_ns);
    break;
  case EventKind::kSourceOwnerDied:
  case EventKind::kSourceInboxOverflow:
    quarantine_live(lifecycle);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestQuarantined,
                      completed_ns);
    break;
  default:
    throw std::runtime_error("decode event routed to source lifecycle");
  }
  verify_conservation(lifecycle);
  return !output.actions.empty();
}

bool reduce_decode_locked(SharedOwner &owner, Lifecycle &lifecycle,
                          const Event &event, Output &output,
                          std::uint64_t completed_ns) {
  const DecodePhase phase = static_cast<DecodePhase>(lifecycle.phase);
  if (phase == DecodePhase::kRetired || phase == DecodePhase::kQuarantined) {
    throw std::runtime_error("decode lifecycle is terminal");
  }
  auto transition = [&](DecodePhase expected, DecodePhase next) {
    if (phase != expected) {
      throw std::runtime_error("decode event is illegal from current phase");
    }
    lifecycle.phase = static_cast<std::uint8_t>(next);
  };
  switch (event.kind) {
  case EventKind::kDecodeAllocationPublished:
    transition(DecodePhase::kPrepared, DecodePhase::kPublished);
    break;
  case EventKind::kDecodeWriterAggregationStarted:
    transition(DecodePhase::kPublished, DecodePhase::kWriterAggregating);
    break;
  case EventKind::kDecodeWriterManifestCompleted:
    transition(DecodePhase::kWriterAggregating, DecodePhase::kScatterReady);
    add_action_locked(owner, lifecycle, output, ActionKind::kDecodeScatterReady,
                      completed_ns);
    break;
  case EventKind::kDecodeScatterStarted:
    transition(DecodePhase::kScatterReady, DecodePhase::kScatterInFlight);
    arm_deadline_locked(owner, lifecycle, DeadlineKind::kOwnerDecodeScatter,
                        completed_ns);
    break;
  case EventKind::kDecodeScatterTerminal:
    transition(DecodePhase::kScatterInFlight, DecodePhase::kScatterTerminal);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerDecodeScatter);
    add_action_locked(owner, lifecycle,
                      output, ActionKind::kDecodeTeardownReady, completed_ns);
    break;
  case EventKind::kDecodeTeardownSent:
    transition(DecodePhase::kScatterTerminal, DecodePhase::kTeardownSent);
    arm_deadline_locked(owner, lifecycle, DeadlineKind::kOwnerTeardownAck,
                        completed_ns);
    break;
  case EventKind::kDecodeAckAggregationStarted:
    transition(DecodePhase::kTeardownSent, DecodePhase::kAckAggregating);
    break;
  case EventKind::kDecodeAckManifestCompleted:
    transition(DecodePhase::kAckAggregating, DecodePhase::kAdoptionReady);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerTeardownAck);
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerSchedulerReceiptConsumption,
                        completed_ns);
    add_action_locked(owner, lifecycle, output, ActionKind::kAdoptionReady,
                      completed_ns, ReceiptKind::kAdoptionReady);
    break;
  case EventKind::kDecodeAdoptionConsumed:
    transition(DecodePhase::kAdoptionReady,
               DecodePhase::kAdoptedByScheduler);
    cancel_deadline_locked(lifecycle,
                           DeadlineKind::kOwnerSchedulerReceiptConsumption);
    break;
  case EventKind::kDecodeMetadataConsumed:
    transition(DecodePhase::kAdoptedByScheduler,
               DecodePhase::kMetadataConsumed);
    break;
  case EventKind::kDecodeLocalReadyIssued:
    transition(DecodePhase::kMetadataConsumed, DecodePhase::kLocalDecodeReady);
    arm_deadline_locked(owner, lifecycle,
                        DeadlineKind::kOwnerRequestGlobalReady, completed_ns);
    add_action_locked(owner, lifecycle, output, ActionKind::kLocalDecodeReady,
                      completed_ns, ReceiptKind::kLocalDecodeReady);
    break;
  case EventKind::kDecodeRequestReady:
    transition(DecodePhase::kLocalDecodeReady, DecodePhase::kRequestReady);
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerRequestGlobalReady);
    lifecycle.live_resources = 0;
    lifecycle.retired_resources = kDecodeResourceMask;
    lifecycle.phase = static_cast<std::uint8_t>(DecodePhase::kRetired);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestRetired,
                      completed_ns);
    break;
  case EventKind::kDecodeCancelUnpublished:
    if (phase != DecodePhase::kPrepared) {
      throw std::runtime_error("published decode allocation cannot roll back");
    }
    lifecycle.live_resources = 0;
    lifecycle.retired_resources = kDecodeResourceMask;
    lifecycle.phase = static_cast<std::uint8_t>(DecodePhase::kRetired);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestRetired,
                      completed_ns);
    break;
  case EventKind::kDecodeRequestFailed:
    if (phase == DecodePhase::kPrepared || lifecycle.live_resources == 0) {
      throw std::runtime_error(
          "decode failure requires a published live allocation");
    }
    quarantine_live(lifecycle);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestQuarantined,
                      completed_ns);
    break;
  case EventKind::kDecodeOwnerDied:
  case EventKind::kDecodeInboxOverflow:
    quarantine_live(lifecycle);
    add_action_locked(owner, lifecycle, output, ActionKind::kRequestQuarantined,
                      completed_ns);
    break;
  default:
    throw std::runtime_error("source event routed to decode lifecycle");
  }
  verify_conservation(lifecycle);
  return !output.actions.empty();
}

void publish_quarantine_for_live_locked(SharedOwner &owner, FatalCode code,
                                        const Event *trigger) {
  const std::uint64_t completed_ns = owner_now_ns_locked(owner);
  const std::size_t live_count = std::count_if(
      owner.lifecycles.begin(), owner.lifecycles.end(),
      [](const auto &entry) { return entry.second.live_resources != 0; });
  if (owner.fatal_output_queue.size() + live_count >
      owner.maximum_live_lifecycles) {
    owner.fatal_code = FatalCode::kInternalError;
    owner.fatal_reason = "fatal output reserve violated lifecycle bound";
    owner.condition.notify_all();
    return;
  }
  for (auto &entry : owner.lifecycles) {
    Lifecycle &lifecycle = entry.second;
    if (lifecycle.live_resources == 0) {
      continue;
    }
    const std::uint8_t previous_phase = lifecycle.phase;
    quarantine_live(lifecycle);
    Output output{};
    output.binding = lifecycle.binding;
    output.owner_sequence = owner.next_owner_sequence++;
    output.producer_id = trigger == nullptr ? 0 : trigger->producer_id;
    output.producer_sequence =
        trigger == nullptr ? 0 : trigger->producer_sequence;
    output.enqueued_ns = trigger == nullptr ? completed_ns : trigger->enqueued_ns;
    output.completed_ns = completed_ns;
    if (trigger != nullptr &&
        ((lifecycle.role == OwnerRole::kSource &&
          static_cast<std::uint16_t>(trigger->kind) < 40) ||
         (lifecycle.role == OwnerRole::kDecode &&
          static_cast<std::uint16_t>(trigger->kind) >= 40))) {
      output.event_kind = trigger->kind;
    } else {
      output.event_kind = lifecycle.role == OwnerRole::kSource
                              ? EventKind::kSourceOwnerDied
                              : EventKind::kDecodeOwnerDied;
    }
    output.role = lifecycle.role;
    output.previous_phase = previous_phase;
    output.phase = lifecycle.phase;
    output.live_resources = lifecycle.live_resources;
    output.retired_resources = lifecycle.retired_resources;
    output.quarantined_resources = lifecycle.quarantined_resources;
    output.armed_deadline_mask = lifecycle.armed_deadline_mask;
    output.process_fatal = true;
    output.fatal_code = code;
    add_action_locked(owner, lifecycle, output, ActionKind::kProcessFatal,
                      completed_ns);
    register_forward_independent_handoffs_locked(owner, output);
    owner.fatal_output_queue.push_back(std::move(output));
  }
  signal_fd_locked(owner, owner.output_fd);
  owner.condition.notify_all();
  owner.qualification_condition.notify_all();
}

void quarantine_all_locked(SharedOwner &owner, FatalCode code,
                           const Event *trigger, const std::string &reason) {
  if (owner.fatal_code != FatalCode::kNone) {
    return;
  }
  owner.fatal_code = code;
  owner.fatal_reason = reason;
  owner.admission_open = false;
  owner.event_admission_open = false;
  if (trigger != nullptr) {
    owner.fatal_binding = trigger->binding_digest;
    owner.fatal_producer_id = trigger->producer_id;
    owner.fatal_producer_sequence = trigger->producer_sequence;
    owner.fatal_reason_code = trigger->reason_code;
    owner.fatal_backend_status = trigger->backend_status;
  }
  publish_quarantine_for_live_locked(owner, code, trigger);
}

bool action_requires_forward_independent_handoff(ActionKind kind) noexcept {
  return kind != ActionKind::kReclaimAuthorized &&
         kind != ActionKind::kAdoptionReady;
}

bool handoff_pending_through_locked(const SharedOwner &owner,
                                    std::uint64_t watermark) noexcept {
  return std::any_of(
      owner.handoff_action_ids.begin(), owner.handoff_action_ids.end(),
      [watermark](std::uint64_t action_id) { return action_id <= watermark; });
}

void enter_handoff_fatal_locked(SharedOwner &owner, FatalCode code,
                                const std::string &reason) noexcept {
  try {
    quarantine_all_locked(owner, code, nullptr, reason);
  } catch (...) {
    if (owner.fatal_code == FatalCode::kNone) {
      owner.fatal_code = code;
      owner.fatal_reason = reason;
      owner.admission_open = false;
      owner.event_admission_open = false;
    }
  }
  owner.condition.notify_all();
  owner.qualification_condition.notify_all();
}

bool schedule_handoff_pending_call_locked(SharedOwner &owner,
                                          std::uint64_t watermark) noexcept {
  if (owner.handoff_callback_scheduled) {
    owner.scheduled_handoff_watermark =
        std::max(owner.scheduled_handoff_watermark, watermark);
    return true;
  }
  try {
    auto call = std::make_unique<PendingHandoffCall>();
    call->owner = owner.shared_from_this();
    call->callback_id = owner.next_handoff_callback_id++;
    if (Py_AddPendingCall(&terminal_handoff_pending_call, call.get()) != 0) {
      enter_handoff_fatal_locked(
          owner, FatalCode::kPendingCallQueueFailure,
          "CPython rejected the terminal handoff pending call");
      return false;
    }
    owner.handoff_callback_scheduled = true;
    owner.scheduled_handoff_callback_id = call->callback_id;
    owner.scheduled_handoff_watermark = watermark;
    call.release();
    return true;
  } catch (...) {
    enter_handoff_fatal_locked(
        owner, FatalCode::kPendingCallQueueFailure,
        "terminal handoff pending-call allocation failed");
    return false;
  }
}

void register_forward_independent_handoffs_locked(
    SharedOwner &owner, const Output &output) noexcept {
  if (!owner.handoff_enabled) {
    return;
  }
  std::uint64_t watermark = 0;
  for (const Action &action : output.actions) {
    if (!action_requires_forward_independent_handoff(action.kind)) {
      continue;
    }
    if (!owner.handoff_action_ids.insert(action.action_id).second) {
      enter_handoff_fatal_locked(
          owner, FatalCode::kHandoffAuthority,
          "terminal handoff registered a duplicate action identity");
      return;
    }
    watermark = std::max(watermark, action.action_id);
  }
  if (watermark == 0 ||
      owner.fatal_code == FatalCode::kPendingCallQueueFailure) {
    return;
  }
  schedule_handoff_pending_call_locked(owner, watermark);
}

void fail_forward_independent_handoff_locked(
    SharedOwner &owner, std::uint64_t action_id) noexcept {
  if (!owner.handoff_enabled) {
    return;
  }
  if (owner.handoff_action_ids.erase(action_id) == 1) {
    ++owner.handoff_completion_count;
  }
  owner.condition.notify_all();
}

int terminal_handoff_pending_call(void *opaque) noexcept {
  std::unique_ptr<PendingHandoffCall> call(
      static_cast<PendingHandoffCall *>(opaque));
  if (call == nullptr || call->owner == nullptr) {
    return 0;
  }
  SharedOwner &owner = *call->owner;
  std::unique_lock<std::mutex> lock(owner.mutex);
  if (owner.closed || !owner.handoff_enabled) {
    return 0;
  }
  if (!owner.handoff_callback_scheduled ||
      owner.scheduled_handoff_callback_id != call->callback_id ||
      owner.handoff_callback_active) {
    enter_handoff_fatal_locked(
        owner, FatalCode::kHandoffAuthority,
        "terminal handoff callback identity was stale or concurrent");
    return 0;
  }
  const std::uint64_t watermark = owner.scheduled_handoff_watermark;
  owner.handoff_callback_scheduled = false;
  owner.scheduled_handoff_callback_id = 0;
  owner.scheduled_handoff_watermark = 0;
  owner.handoff_callback_active = true;
  owner.active_handoff_callback_id = call->callback_id;
  owner.active_handoff_watermark = watermark;
  ++owner.handoff_callback_count;
  owner.condition.notify_all();

  const DeadlineSpec &shutdown = owner.deadline_specs[
      static_cast<std::uint8_t>(DeadlineKind::kOwnerShutdownDrain)];
  const std::uint64_t started_ns = owner_now_ns_locked(owner);
  const std::uint64_t deadline_ns = started_ns + shutdown.duration_ns;
  PyThreadState *thread_state = PyEval_SaveThread();
  bool wait_failed = false;
  bool completed = false;
  bool deadline_expired = false;
  try {
    completed = owner.condition.wait_for(
        lock, std::chrono::nanoseconds(shutdown.duration_ns), [&]() {
          if (!handoff_pending_through_locked(owner, watermark) ||
              owner.closed) {
            return true;
          }
#ifdef SGLANG_TERMINAL_OWNER_TESTING
          return owner.test_clock_enabled &&
                 owner_now_ns_locked(owner) >= deadline_ns;
#else
          return false;
#endif
        });
    deadline_expired =
        handoff_pending_through_locked(owner, watermark) && !owner.closed &&
        owner_now_ns_locked(owner) >= deadline_ns;
  } catch (...) {
    wait_failed = true;
  }
  if (wait_failed) {
    enter_handoff_fatal_locked(
        owner, FatalCode::kHandoffAuthority,
        "terminal handoff callback condition wait failed");
  } else if (!completed || deadline_expired) {
    enter_handoff_fatal_locked(
        owner, FatalCode::kHandoffTimeout,
        "terminal handoff callback exceeded owner shutdown deadline");
  }
  owner.handoff_callback_active = false;
  owner.active_handoff_callback_id = 0;
  owner.active_handoff_watermark = 0;
  owner.condition.notify_all();
  lock.unlock();
  PyEval_RestoreThread(thread_state);
  return 0;
}

void apply_publication_owner_loss_locked(Lifecycle &lifecycle) {
  const SourcePhase source_phase = static_cast<SourcePhase>(lifecycle.phase);
  const bool publication_already_terminal =
      lifecycle.role == OwnerRole::kSource &&
      lifecycle.publication_authorized &&
      (lifecycle.gateway_published || lifecycle.publication_quarantined);
  const bool decode_adoption_proven =
      lifecycle.role == OwnerRole::kSource &&
      (source_phase == SourcePhase::kRequestReadyReceived ||
       source_phase == SourcePhase::kPublicationQuarantined) &&
      lifecycle.publication_authorized && !lifecycle.gateway_published;
  if (publication_already_terminal) {
    // Decode has adopted its pages, and the publication identity already has
    // an exact terminal disposition. Losing its owner changes only process
    // liveness; it cannot make proven source storage ambiguous again.
    return;
  }
  if (!decode_adoption_proven) {
    quarantine_live(lifecycle);
    return;
  }
  if (!lifecycle.publication_quarantined) {
    lifecycle.live_resources &= ~(1ULL << 8);
    lifecycle.quarantined_resources |= 1ULL << 8;
    lifecycle.publication_quarantined = true;
    cancel_deadline_locked(lifecycle, DeadlineKind::kOwnerGatewayPublication);
  }
  lifecycle.phase =
      static_cast<std::uint8_t>(SourcePhase::kPublicationQuarantined);
}

void publication_owner_failure_locked(SharedOwner &owner, FatalCode code,
                                      const Event &trigger,
                                      const std::string &reason) {
  if (owner.fatal_code != FatalCode::kNone) {
    return;
  }
  owner.fatal_code = code;
  owner.fatal_reason = reason;
  owner.admission_open = false;
  owner.event_admission_open = false;
  owner.fatal_binding = trigger.binding_digest;
  owner.fatal_producer_id = trigger.producer_id;
  owner.fatal_producer_sequence = trigger.producer_sequence;
  owner.fatal_reason_code = trigger.reason_code;
  owner.fatal_backend_status = trigger.backend_status;
  const std::uint64_t completed_ns = owner_now_ns_locked(owner);
  const std::size_t live_count = std::count_if(
      owner.lifecycles.begin(), owner.lifecycles.end(),
      [](const auto &entry) { return entry.second.live_resources != 0; });
  if (owner.fatal_output_queue.size() + live_count >
      owner.maximum_live_lifecycles) {
    owner.fatal_code = FatalCode::kInternalError;
    owner.fatal_reason = "fatal output reserve violated lifecycle bound";
    owner.condition.notify_all();
    return;
  }
  for (auto &entry : owner.lifecycles) {
    Lifecycle &lifecycle = entry.second;
    if (lifecycle.live_resources == 0) {
      continue;
    }
    const std::uint8_t previous_phase = lifecycle.phase;
    apply_publication_owner_loss_locked(lifecycle);
    verify_conservation(lifecycle);
    Output output{};
    output.binding = lifecycle.binding;
    output.owner_sequence = owner.next_owner_sequence++;
    output.producer_id = trigger.producer_id;
    output.producer_sequence = trigger.producer_sequence;
    output.enqueued_ns = trigger.enqueued_ns;
    output.completed_ns = completed_ns;
    output.event_kind = trigger.kind;
    output.role = lifecycle.role;
    output.previous_phase = previous_phase;
    output.phase = lifecycle.phase;
    output.live_resources = lifecycle.live_resources;
    output.retired_resources = lifecycle.retired_resources;
    output.quarantined_resources = lifecycle.quarantined_resources;
    output.armed_deadline_mask = lifecycle.armed_deadline_mask;
    output.process_fatal = true;
    output.fatal_code = owner.fatal_code;
    add_action_locked(owner, lifecycle, output, ActionKind::kProcessFatal,
                      completed_ns);
    register_forward_independent_handoffs_locked(owner, output);
    owner.fatal_output_queue.push_back(std::move(output));
  }
  signal_fd_locked(owner, owner.output_fd);
  owner.condition.notify_all();
  owner.qualification_condition.notify_all();
}

void publisher_death_locked(SharedOwner &owner, const Event &event) {
  publication_owner_failure_locked(
      owner, FatalCode::kDependencyDeath, event,
      event.reason.empty() ? "terminal publisher died" : event.reason);
}

void configure_timer_locked(SharedOwner &owner) {
#ifdef SGLANG_TERMINAL_OWNER_TESTING
  if (owner.test_clock_enabled) {
    return;
  }
#endif
  std::uint64_t nearest = 0;
  for (const auto &entry : owner.lifecycles) {
    const Lifecycle &lifecycle = entry.second;
    for (std::uint8_t index = 1; index < lifecycle.deadline_expiry_ns.size();
         ++index) {
      const std::uint64_t expiry = lifecycle.deadline_expiry_ns[index];
      if (expiry != 0 && (nearest == 0 || expiry < nearest)) {
        nearest = expiry;
      }
    }
  }
  itimerspec spec{};
  if (nearest != 0) {
    const std::uint64_t now_ns = owner_now_ns_locked(owner);
    const std::uint64_t delay_ns = nearest > now_ns ? nearest - now_ns : 1;
    spec.it_value.tv_sec = static_cast<time_t>(delay_ns / 1'000'000'000ULL);
    spec.it_value.tv_nsec = static_cast<long>(delay_ns % 1'000'000'000ULL);
  }
  if (timerfd_settime(owner.timer_fd, 0, &spec, nullptr) != 0) {
    quarantine_all_locked(owner, FatalCode::kTimerfdFailure, nullptr,
                          "timerfd_settime failed");
  }
}

void expire_deadlines_locked(SharedOwner &owner) {
  const std::uint64_t now_ns = owner_now_ns_locked(owner);
  for (auto &entry : owner.lifecycles) {
    Lifecycle &lifecycle = entry.second;
    if (lifecycle_terminal(lifecycle)) {
      continue;
    }
    for (std::uint8_t index = 1; index < lifecycle.deadline_expiry_ns.size();
         ++index) {
      const std::uint64_t expiry = lifecycle.deadline_expiry_ns[index];
      if (expiry == 0 || expiry > now_ns) {
        continue;
      }
      const DeadlineSpec &spec = owner.deadline_specs[index];
      if (spec.process_fatal) {
        Event trigger{};
        trigger.binding_digest = lifecycle.binding.digest;
        trigger.enqueued_ns = expiry;
        if (spec.kind == DeadlineKind::kOwnerGatewayPublication) {
          publication_owner_failure_locked(
              owner, FatalCode::kDeadlineExpiry, trigger,
              "process-fatal owner deadline expired");
        } else {
          quarantine_all_locked(owner, FatalCode::kDeadlineExpiry, &trigger,
                                "process-fatal owner deadline expired");
        }
        return;
      }
      const std::uint8_t previous_phase = lifecycle.phase;
      quarantine_live(lifecycle);
      Output output{};
      output.binding = lifecycle.binding;
      output.owner_sequence = owner.next_owner_sequence++;
      output.enqueued_ns = expiry;
      output.completed_ns = now_ns;
      output.event_kind = lifecycle.role == OwnerRole::kSource
                              ? EventKind::kSourceRequestFailed
                              : EventKind::kDecodeRequestFailed;
      output.role = lifecycle.role;
      output.previous_phase = previous_phase;
      output.phase = lifecycle.phase;
      output.live_resources = lifecycle.live_resources;
      output.retired_resources = lifecycle.retired_resources;
      output.quarantined_resources = lifecycle.quarantined_resources;
      output.armed_deadline_mask = lifecycle.armed_deadline_mask;
      output.fatal_code = FatalCode::kNone;
      add_action_locked(owner, lifecycle, output,
                        ActionKind::kRequestQuarantined, now_ns);
      if (owner.output_queue.size() >= owner.output_capacity) {
        if (owner.fatal_output_queue.size() >=
            owner.maximum_live_lifecycles) {
          discard_unpublished_actions_locked(owner, output);
          quarantine_all_locked(owner, FatalCode::kInternalError, nullptr,
                                "deadline fatal-output reserve overflowed");
          return;
        }
        register_forward_independent_handoffs_locked(owner, output);
        owner.fatal_output_queue.push_back(std::move(output));
        quarantine_all_locked(owner, FatalCode::kOutputQueueOverflow, nullptr,
                              "deadline output overflowed");
        return;
      }
      register_forward_independent_handoffs_locked(owner, output);
      owner.output_queue.push_back(std::move(output));
      signal_fd_locked(owner, owner.output_fd);
      break;
    }
  }
  configure_timer_locked(owner);
}

bool qualification_binding_locked(const SharedOwner &owner,
                                  const Digest &digest,
                                  std::size_t *index) {
  for (std::size_t current = 0;
       current < owner.qualification.bindings.size(); ++current) {
    if (owner.qualification.bindings[current] == digest) {
      *index = current;
      return true;
    }
  }
  return false;
}

EventKind qualification_event_kind(std::uint8_t hop) {
  static constexpr std::array<EventKind, kQualificationLifecycleHopCount> kinds{
      EventKind::kSourceSubmissionAccepted,
      EventKind::kSourceProducerCompleted,
      EventKind::kSourceGatherPosted,
      EventKind::kSourceNativeTerminal,
      EventKind::kSourceOutcomesSent,
      EventKind::kSourceTeardownReceived,
      EventKind::kSourceAckSent,
      EventKind::kSourceRequestReady,
      EventKind::kSourceReclaimConsumed,
      EventKind::kSourceGatewayPublished,
  };
  return kinds.at(hop);
}

std::uint64_t nearest_rank_from_sorted(
    const std::vector<std::uint64_t> &values, std::uint64_t percentile) {
  if (values.empty() || percentile == 0 || percentile > 100) {
    throw std::runtime_error("qualification percentile input is invalid");
  }
  const std::uint64_t count = values.size();
  const std::uint64_t rank = (percentile * count + 99) / 100;
  return values.at(static_cast<std::size_t>(rank - 1));
}

LatencyStatistics summarize_latencies(std::vector<std::uint64_t> &values) {
  if (values.empty()) {
    throw std::runtime_error("qualification latency population is empty");
  }
  std::sort(values.begin(), values.end());
  LatencyStatistics result{};
  result.count = values.size();
  result.p50_ns = nearest_rank_from_sorted(values, 50);
  result.p95_ns = nearest_rank_from_sorted(values, 95);
  result.p99_ns = nearest_rank_from_sorted(values, 99);
  result.maximum_ns = values.back();
  return result;
}

void record_qualification_trace_locked(QualificationState &qualification,
                                       const Trace &trace,
                                       std::size_t machine_index,
                                       std::uint8_t hop) {
  if (hop == 0 && qualification.path_latencies.size() >=
                      kQualificationMaximumFullPathSampleCount) {
    throw std::runtime_error(
        "qualification statistics capacity was exhausted");
  }
  if (trace.completed_ns < trace.event.enqueued_ns) {
    throw std::runtime_error("qualification completion preceded enqueue");
  }
  const std::uint64_t latency_ns =
      trace.completed_ns - trace.event.enqueued_ns;
  Trace measured = trace;
  measured.machine_index = machine_index;
  measured.generation_index = qualification.generations.at(machine_index);
  measured.hop_index = hop;
  auto &first = qualification.first_audit_traces.at(machine_index).at(hop);
  if (!first.has_value()) {
    first = measured;
  }
  qualification.last_audit_traces.at(machine_index).at(hop) = measured;
  qualification.hop_latencies.at(hop).push_back(latency_ns);
  qualification.path_latency_accumulators.at(machine_index) += latency_ns;
  if (hop + 1 == kQualificationMeasuredHopCount) {
    qualification.path_latencies.push_back(
        qualification.path_latency_accumulators.at(machine_index));
    qualification.path_latency_accumulators.at(machine_index) = 0;
  }
  ++qualification.transition_count;
}

void finalize_qualification_summary_locked(QualificationState &qualification) {
  if (qualification.summary_complete) {
    throw std::runtime_error("qualification summary was already finalized");
  }
  if (qualification.transition_count !=
      qualification.path_latencies.size() * kQualificationMeasuredHopCount) {
    throw std::runtime_error(
        "qualification measured transitions do not conserve full paths");
  }
  if (qualification.lifecycle_transition_count !=
      qualification.path_latencies.size() * kQualificationLifecycleHopCount) {
    throw std::runtime_error(
        "qualification lifecycle transitions do not conserve generations");
  }
  if (qualification.owner_sequence_end < qualification.owner_sequence_start ||
      qualification.owner_sequence_end - qualification.owner_sequence_start !=
          qualification.lifecycle_transition_count) {
    throw std::runtime_error(
        "qualification owner sequence does not conserve lifecycle commits");
  }
  std::uint64_t completed_generation_count = 0;
  for (const std::uint64_t count : qualification.completed_generations) {
    completed_generation_count += count;
  }
  if (completed_generation_count != qualification.path_latencies.size()) {
    throw std::runtime_error(
        "qualification per-machine generations do not conserve full paths");
  }
  for (const std::uint64_t accumulator :
       qualification.path_latency_accumulators) {
    if (accumulator != 0) {
      throw std::runtime_error(
          "qualification retired with a partial measured path");
    }
  }
  for (std::size_t hop = 0; hop < kQualificationMeasuredHopCount; ++hop) {
    if (qualification.hop_latencies.at(hop).size() !=
        qualification.path_latencies.size()) {
      throw std::runtime_error(
          "qualification transition classes have unequal populations");
    }
    qualification.hop_statistics.at(hop) =
        summarize_latencies(qualification.hop_latencies.at(hop));
  }
  qualification.path_statistics =
      summarize_latencies(qualification.path_latencies);
  qualification.sample_count = qualification.path_latencies.size();
  for (auto &latencies : qualification.hop_latencies) {
    std::vector<std::uint64_t>().swap(latencies);
  }
  std::vector<std::uint64_t>().swap(qualification.path_latencies);
  qualification.summary_complete = true;
}

void finish_qualification_transition_locked(SharedOwner &owner,
                                            const Event &event,
                                            const Trace &trace,
                                            Output &output) {
  QualificationState &qualification = owner.qualification;
  if (!qualification.running) {
    return;
  }
  std::size_t machine_index = 0;
  if (!qualification_binding_locked(owner, event.binding_digest,
                                    &machine_index)) {
    return;
  }
  for (const Action &action : output.actions) {
    const auto pending = owner.pending_actions.find(action.action_id);
    if (pending == owner.pending_actions.end()) {
      throw std::runtime_error("qualification action lost native authority");
    }
    owner.pending_actions.erase(pending);
    owner.consumed_actions.insert(action.action_id);
  }
  // Qualification's native synthetic consumer acknowledges actions here so
  // Python GIL starvation cannot turn an owner-latency test into an output-
  // consumer test. The actions were minted by the production reducer and pass
  // through the same replay ledger before the test-only trace is retained.
  output.actions.clear();
  std::uint8_t &hop = qualification.hops[machine_index];
  ++qualification.lifecycle_transition_count;
  if (hop < kQualificationMeasuredHopCount) {
    record_qualification_trace_locked(qualification, trace, machine_index, hop);
  }
  ++hop;
  const bool closure_complete = hop == kQualificationLifecycleHopCount;
  if (!closure_complete) {
    return;
  }
  ++qualification.completed_generations[machine_index];
  const std::uint64_t now_ns = trace.completed_ns;
  if (!qualification.draining &&
      now_ns - qualification.started_ns >= qualification.minimum_duration_ns &&
      qualification.transition_count >= qualification.minimum_transition_count) {
    qualification.draining = true;
  }
  if (qualification.draining) {
    hop = 255;
    bool all_complete = true;
    for (const std::uint8_t candidate : qualification.hops) {
      if (candidate != 255) {
        all_complete = false;
        break;
      }
    }
    if (all_complete) {
      qualification.ended_ns = now_ns;
      qualification.owner_sequence_end = owner.next_owner_sequence;
      finalize_qualification_summary_locked(qualification);
      qualification.complete = true;
      qualification.running = false;
      for (const std::uint64_t producer_id : qualification.producer_ids) {
        owner.producers.at(producer_id).retirement_requested = true;
        owner.producers.at(producer_id).retired = true;
      }
      owner.qualification_condition.notify_all();
    }
    return;
  }
  ++qualification.generations[machine_index];
  RequestBinding old_binding = owner.lifecycles.at(event.binding_digest).binding;
  owner.lifecycles.erase(event.binding_digest);
  RequestBinding next_binding = old_binding;
  const std::uint64_t generation = qualification.generations[machine_index];
  for (std::size_t index = 0; index < 8; ++index) {
    next_binding.request_generation[index] =
        static_cast<std::uint8_t>(generation >> (index * 8));
    next_binding.digest[index] =
        static_cast<std::uint8_t>((generation ^ machine_index) >> (index * 8));
  }
  next_binding.digest[31] = static_cast<std::uint8_t>(machine_index);
  Lifecycle lifecycle{};
  lifecycle.binding = next_binding;
  lifecycle.role = OwnerRole::kSource;
  lifecycle.phase = static_cast<std::uint8_t>(SourcePhase::kFrozen);
  lifecycle.live_resources = kSourceResourceMask;
  lifecycle.trusted_issuers.insert(owner.owner_identity.digest);
  owner.lifecycles.emplace(next_binding.digest, lifecycle);
  qualification.bindings[machine_index] = next_binding.digest;
  hop = 0;
}

void enqueue_next_qualification_events_locked(SharedOwner &owner) {
  QualificationState &qualification = owner.qualification;
  if (!qualification.running || owner.stop_requested ||
      owner.fatal_code != FatalCode::kNone) {
    return;
  }
  for (std::size_t index = 0; index < qualification.machine_count; ++index) {
    const std::uint8_t hop = qualification.hops[index];
    if (hop == 255) {
      continue;
    }
    const std::uint64_t sequence = qualification.producer_sequences[index];
    bool already_queued = false;
    for (const InputCommand &queued : owner.input_queue) {
      if (queued.kind == InputKind::kEvent &&
          queued.event.producer_id == qualification.producer_ids[index] &&
          queued.event.producer_sequence == sequence) {
        already_queued = true;
        break;
      }
    }
    if (already_queued) {
      continue;
    }
    Event event{};
    event.producer_id = qualification.producer_ids[index];
    event.producer_sequence = sequence;
    event.binding_digest = qualification.bindings[index];
    event.kind = qualification_event_kind(hop);
    event.enqueued_ns = owner_now_ns_locked(owner);
    if (event_requires_receipt(event.kind)) {
      Lifecycle &lifecycle = owner.lifecycles.at(event.binding_digest);
      event.has_receipt = true;
      event.receipt = mint_receipt_locked(
          owner, lifecycle.binding, expected_receipt(event.kind).first,
          expected_receipt(event.kind).second, event.enqueued_ns);
    }
    if (owner.input_queue.size() >= owner.input_capacity) {
      quarantine_all_locked(owner, FatalCode::kInputQueueOverflow, &event,
                            "qualification input queue overflowed");
      return;
    }
    InputCommand command{};
    command.kind = InputKind::kEvent;
    command.event = std::move(event);
    owner.input_queue.push_back(std::move(command));
    ++qualification.producer_sequences[index];
    owner.producers.at(qualification.producer_ids[index])
        .next_submission_sequence = qualification.producer_sequences[index];
  }
  signal_fd_locked(owner, owner.input_fd);
}

void register_lifecycle_locked(SharedOwner &owner, Lifecycle lifecycle) {
  const auto existing = owner.lifecycles.find(lifecycle.binding.digest);
  if (existing == owner.lifecycles.end()) {
    owner.lifecycles.emplace(lifecycle.binding.digest, std::move(lifecycle));
    owner.condition.notify_all();
    return;
  }
  const std::string reason =
      same_binding(existing->second.binding, lifecycle.binding)
          ? "binding was already registered"
          : "binding digest collision changed full payload";
  Event trigger{};
  trigger.binding_digest = lifecycle.binding.digest;
  trigger.enqueued_ns = owner_now_ns_locked(owner);
  quarantine_all_locked(owner, FatalCode::kDuplicateBinding, &trigger, reason);
}

void observe_submission_commit_locked(SharedOwner &owner,
                                      const Output &output) noexcept {
  if (output.event_kind != EventKind::kSourceSubmissionAccepted ||
      !output.actions.empty()) {
    return;
  }
  ++owner.observation_count;
  if (owner.observation_queue.size() >= owner.observation_capacity) {
    ++owner.dropped_observation_count;
    return;
  }
  try {
    Observation observation{};
    observation.binding = output.binding;
    observation.owner_sequence = output.owner_sequence;
    observation.producer_id = output.producer_id;
    observation.producer_sequence = output.producer_sequence;
    observation.producer_rank = output.binding.owner.tp_rank;
    observation.event_kind = output.event_kind;
    observation.enqueued_ns = output.enqueued_ns;
    observation.completed_ns = output.completed_ns;
    observation.role = output.role;
    owner.observation_queue.push_back(std::move(observation));
  } catch (...) {
    ++owner.dropped_observation_count;
    return;
  }
  if (!signal_observation_fd_locked(owner)) {
    owner.observation_queue.pop_back();
    ++owner.dropped_observation_count;
  }
}

void dispatch_event_locked(SharedOwner &owner, const Event &event) {
  const auto producer_iterator = owner.producers.find(event.producer_id);
  if (producer_iterator == owner.producers.end()) {
    quarantine_all_locked(owner, FatalCode::kProducerSequence, &event,
                          "event used an unknown producer");
    return;
  }
  ProducerRegistration &producer = producer_iterator->second;
  if (producer.retired ||
      event.producer_sequence != producer.next_dispatch_sequence) {
    quarantine_all_locked(
        owner, FatalCode::kProducerSequence, &event,
        producer.retired ? "event followed committed producer retirement"
                         : "owner-assigned producer sequence was not gap-free");
    return;
  }
  ++producer.next_dispatch_sequence;
  if (!producer_authorizes_event(producer, event)) {
    quarantine_all_locked(owner, FatalCode::kReceiptAuthority, &event,
                          "producer class does not authorize event kind");
    return;
  }
  const auto lifecycle_iterator = owner.lifecycles.find(event.binding_digest);
  if (lifecycle_iterator == owner.lifecycles.end()) {
    quarantine_all_locked(owner, FatalCode::kUnknownBinding, &event,
                          "event targeted an unknown binding digest");
    return;
  }
  Lifecycle &lifecycle = lifecycle_iterator->second;
  if (lifecycle.binding.digest != event.binding_digest ||
      lifecycle.role != producer.allowed_role) {
    quarantine_all_locked(owner, FatalCode::kUnknownBinding, &event,
                          "binding identity or producer role mismatched");
    return;
  }
  const bool process_fatal_event =
      event.kind == EventKind::kSourceOwnerDied ||
      event.kind == EventKind::kSourceInboxOverflow ||
      event.kind == EventKind::kDecodeOwnerDied ||
      event.kind == EventKind::kDecodeInboxOverflow;
  if (process_fatal_event) {
    quarantine_all_locked(owner, FatalCode::kDependencyDeath, &event,
                          event.reason.empty()
                              ? "terminal owner dependency died"
                              : event.reason);
    return;
  }
  if (event.kind == EventKind::kSourcePublisherDied) {
    publisher_death_locked(owner, event);
    return;
  }
  try {
    Event canonical_event = event;
    const bool local_failure =
        producer.producer_class == ProducerClass::kLocal &&
        (event.kind == EventKind::kSourceRequestFailed ||
         event.kind == EventKind::kDecodeRequestFailed);
    if (local_failure) {
      canonical_event.has_receipt = true;
      canonical_event.receipt = mint_receipt_locked(
          owner, lifecycle.binding, ReceiptKind::kFailure,
          ReceiptOutcome::kFailure, owner_now_ns_locked(owner));
      ProducerRegistration owner_producer = producer;
      owner_producer.producer_class = ProducerClass::kReceipt;
      owner_producer.has_issuer = true;
      owner_producer.issuer = owner.owner_identity;
      lifecycle.trusted_issuers.insert(owner.owner_identity.digest);
      validate_receipt_locked(lifecycle, owner_producer, canonical_event);
    } else {
      validate_receipt_locked(lifecycle, producer, canonical_event);
    }
    const std::uint8_t previous_phase = lifecycle.phase;
    const std::uint64_t completed_ns = owner_now_ns_locked(owner);
    Output output{};
    output.binding = lifecycle.binding;
    output.owner_sequence = owner.next_owner_sequence++;
    output.producer_id = event.producer_id;
    output.producer_sequence = event.producer_sequence;
    output.enqueued_ns = event.enqueued_ns;
    output.completed_ns = completed_ns;
    output.event_kind = canonical_event.kind;
    output.role = lifecycle.role;
    output.previous_phase = previous_phase;
    if (lifecycle.role == OwnerRole::kSource) {
      reduce_source_locked(owner, lifecycle, canonical_event, output,
                           completed_ns);
    } else {
      reduce_decode_locked(owner, lifecycle, canonical_event, output,
                           completed_ns);
    }
    output.phase = lifecycle.phase;
    output.live_resources = lifecycle.live_resources;
    output.retired_resources = lifecycle.retired_resources;
    output.quarantined_resources = lifecycle.quarantined_resources;
    output.armed_deadline_mask = lifecycle.armed_deadline_mask;
    output.process_fatal = owner.fatal_code != FatalCode::kNone;
    output.fatal_code = owner.fatal_code;
    Trace trace{canonical_event, completed_ns, output.owner_sequence,
                previous_phase,
                lifecycle.phase};
    finish_qualification_transition_locked(owner, canonical_event, trace,
                                            output);
    observe_submission_commit_locked(owner, output);
    if (!output.actions.empty()) {
      if (owner.output_queue.size() >= owner.output_capacity) {
        discard_unpublished_actions_locked(owner, output);
        quarantine_all_locked(owner, FatalCode::kOutputQueueOverflow, &event,
                              "production action queue overflowed");
        return;
      }
      register_forward_independent_handoffs_locked(owner, output);
      owner.output_queue.push_back(std::move(output));
      signal_fd_locked(owner, owner.output_fd);
    }
    configure_timer_locked(owner);
  } catch (const std::exception &error) {
    const std::string message = error.what();
    FatalCode code = FatalCode::kIllegalTransition;
    if (message.find("receipt") != std::string::npos) {
      code = message.find("replayed") != std::string::npos
                 ? FatalCode::kReceiptReplay
                 : FatalCode::kReceiptAuthority;
    }
    quarantine_all_locked(owner, code, &event, message);
  }
}

void retire_producer_locked(SharedOwner &owner, const InputCommand &command) {
  const auto producer_iterator = owner.producers.find(command.producer_id);
  if (producer_iterator == owner.producers.end()) {
    quarantine_all_locked(owner, FatalCode::kInternalError, nullptr,
                          "retirement targeted an unknown producer");
    return;
  }
  ProducerRegistration &producer = producer_iterator->second;
  const bool ordered =
      producer.retirement_requested && !producer.retired &&
      producer.next_submission_sequence == command.retire_after_sequence &&
      producer.next_dispatch_sequence == command.retire_after_sequence;
  if (!ordered) {
    quarantine_all_locked(
        owner, FatalCode::kInternalError, nullptr,
        "producer retirement crossed its accepted-event fence");
    return;
  }
  producer.retired = true;
  owner.condition.notify_all();
}

void finish_producer_join_barrier_locked(SharedOwner &owner) {
  if (!owner.producer_join_requested ||
      owner.producer_join_barrier_complete || !owner.input_queue.empty()) {
    return;
  }
  owner.producers_joined = std::all_of(
      owner.producers.begin(), owner.producers.end(),
      [](const auto &entry) { return entry.second.retired; });
  owner.producer_join_barrier_complete = true;
  owner.condition.notify_all();
}

void reactor_main(std::shared_ptr<SharedOwner> owner) noexcept {
  pollfd descriptors[3]{{owner->input_fd, POLLIN, 0},
                        {owner->timer_fd, POLLIN, 0},
                        {owner->shutdown_fd, POLLIN, 0}};
  for (;;) {
    const int result = ::poll(descriptors, 3, -1);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::lock_guard<std::mutex> lock(owner->mutex);
      quarantine_all_locked(*owner, FatalCode::kEventfdFailure, nullptr,
                            "owner poll failed");
      return;
    }
    if ((descriptors[2].revents & POLLIN) != 0) {
      const int error = consume_fd(owner->shutdown_fd);
      std::lock_guard<std::mutex> lock(owner->mutex);
      if (error != 0) {
        owner->fatal_system_error = error;
        quarantine_all_locked(*owner, FatalCode::kEventfdFailure, nullptr,
                              "shutdown eventfd read failed");
        return;
      }
      if (owner->stop_requested && owner->input_queue.empty()) {
        finish_producer_join_barrier_locked(*owner);
        owner->condition.notify_all();
        return;
      }
    }
    if ((descriptors[1].revents & POLLIN) != 0) {
      const int error = consume_timer_fd(owner->timer_fd);
      std::lock_guard<std::mutex> lock(owner->mutex);
      if (error != 0) {
        owner->fatal_system_error = error;
        quarantine_all_locked(*owner, FatalCode::kTimerfdFailure, nullptr,
                              "timerfd read failed");
        return;
      }
      expire_deadlines_locked(*owner);
    }
    if ((descriptors[0].revents & POLLIN) == 0) {
      continue;
    }
    const int input_error = consume_fd(owner->input_fd);
    if (input_error != 0) {
      std::lock_guard<std::mutex> lock(owner->mutex);
      owner->fatal_system_error = input_error;
      quarantine_all_locked(*owner, FatalCode::kEventfdFailure, nullptr,
                            "input eventfd read failed");
      return;
    }
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->input_queue.empty()) {
          enqueue_next_qualification_events_locked(*owner);
          if (owner->input_queue.empty()) {
            finish_producer_join_barrier_locked(*owner);
            break;
          }
        }
        InputCommand command = std::move(owner->input_queue.front());
        owner->input_queue.pop_front();
        if (owner->fatal_code != FatalCode::kNone &&
            command.kind != InputKind::kRetireProducer) {
          continue;
        }
        if (command.kind == InputKind::kRegisterLifecycle) {
          register_lifecycle_locked(*owner, std::move(command.lifecycle));
        } else if (command.kind == InputKind::kEvent) {
          dispatch_event_locked(*owner, command.event);
        } else {
          retire_producer_locked(*owner, command);
        }
        owner->condition.notify_all();
      }
    }
  }
}

int submit_event_locked(SharedOwner &owner, Event event) noexcept {
  std::lock_guard<std::mutex> lock(owner.mutex);
  if (owner.closed || !owner.event_admission_open ||
      owner.fatal_code != FatalCode::kNone) {
    return ESHUTDOWN;
  }
  const auto producer_iterator = owner.producers.find(event.producer_id);
  if (producer_iterator == owner.producers.end()) {
    return ENOENT;
  }
  ProducerRegistration &producer = producer_iterator->second;
  if (producer.retirement_requested || producer.retired) {
    return ESHUTDOWN;
  }
  if (owner.input_queue.size() >= owner.input_capacity) {
    quarantine_all_locked(owner, FatalCode::kInputQueueOverflow, &event,
                          "native producer input queue overflowed");
    return ENOBUFS;
  }
  event.producer_sequence = producer.next_submission_sequence;
  InputCommand command{};
  command.kind = InputKind::kEvent;
  command.event = std::move(event);
  owner.input_queue.push_back(std::move(command));
  ++producer.next_submission_sequence;
  signal_fd_locked(owner, owner.input_fd);
  return owner.fatal_code == FatalCode::kNone ? 0 : EIO;
}

int submit_producer_retirement_locked(SharedOwner &owner,
                                      std::uint64_t producer_id) noexcept {
  std::lock_guard<std::mutex> lock(owner.mutex);
  if (owner.closed) {
    return ESHUTDOWN;
  }
  const auto producer = owner.producers.find(producer_id);
  if (producer == owner.producers.end()) {
    return ENOENT;
  }
  if (producer->second.retired) {
    return EALREADY;
  }
  if (owner.abort_started) {
    producer->second.retirement_requested = true;
    producer->second.retired = true;
    owner.condition.notify_all();
    return 0;
  }
  if (producer->second.retirement_requested) {
    return EALREADY;
  }
  if (!owner.event_admission_open || owner.fatal_code != FatalCode::kNone) {
    return ESHUTDOWN;
  }
  if (owner.input_queue.size() >= owner.input_capacity) {
    quarantine_all_locked(owner, FatalCode::kInputQueueOverflow, nullptr,
                          "producer retirement queue overflowed");
    return ENOBUFS;
  }
  producer->second.retirement_requested = true;
  InputCommand command{};
  command.kind = InputKind::kRetireProducer;
  command.producer_id = producer_id;
  command.retire_after_sequence = producer->second.next_submission_sequence;
  owner.input_queue.push_back(std::move(command));
  signal_fd_locked(owner, owner.input_fd);
  return owner.fatal_code == FatalCode::kNone ? 0 : EIO;
}

int submit_lifecycle_locked(SharedOwner &owner, Lifecycle lifecycle) noexcept {
  std::lock_guard<std::mutex> lock(owner.mutex);
  if (owner.closed || !owner.admission_open ||
      owner.fatal_code != FatalCode::kNone) {
    return ESHUTDOWN;
  }
  if (owner.input_queue.size() >= owner.input_capacity) {
    Event trigger{};
    trigger.binding_digest = lifecycle.binding.digest;
    quarantine_all_locked(owner, FatalCode::kInputQueueOverflow, &trigger,
                          "native lifecycle registration queue overflowed");
    return ENOBUFS;
  }
  std::size_t pending_registrations = 0;
  for (const InputCommand &command : owner.input_queue) {
    if (command.kind == InputKind::kRegisterLifecycle) {
      ++pending_registrations;
    }
  }
  const std::size_t active_lifecycles = std::count_if(
      owner.lifecycles.begin(), owner.lifecycles.end(),
      [](const auto &entry) { return entry.second.live_resources != 0; });
  if (active_lifecycles + pending_registrations >=
      owner.maximum_live_lifecycles) {
    return ENOSPC;
  }
  InputCommand command{};
  command.kind = InputKind::kRegisterLifecycle;
  command.lifecycle = std::move(lifecycle);
  owner.input_queue.push_back(std::move(command));
  signal_fd_locked(owner, owner.input_fd);
  return owner.fatal_code == FatalCode::kNone ? 0 : EIO;
}

template <std::size_t Size>
bool all_zero(const std::uint8_t (&value)[Size]) noexcept {
  return std::all_of(std::begin(value), std::end(value),
                     [](std::uint8_t byte) { return byte == 0; });
}

int producer_capsule_submit(
    void *context,
    const sglang_terminal_owner_producer_event_v1 *value) noexcept {
  if (context == nullptr || value == nullptr ||
      value->abi_version != SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION ||
      value->struct_size < sizeof(sglang_terminal_owner_producer_event_v1) ||
      value->has_receipt > 1 || !all_zero(value->reserved_after_event_kind) ||
      !all_zero(value->reserved_after_reason_code) ||
      !all_zero(value->reserved_before_receipt_identity)) {
    return EINVAL;
  }
  ProducerCapsule *capsule = static_cast<ProducerCapsule *>(context);
  const std::shared_ptr<SharedOwner> owner = capsule->owner;
  if (owner == nullptr) {
    return ESHUTDOWN;
  }
  Event event{};
  event.producer_id = capsule->producer_id;
  std::memcpy(event.binding_digest.data(), value->binding_digest,
              kDigestBytes);
  event.kind = static_cast<EventKind>(value->event_kind);
  event.enqueued_ns = value->enqueued_ns;
  event.reason_code = value->reason_code;
  event.backend_status = value->backend_status;
  if (value->has_receipt != 0) {
    event.has_receipt = true;
    std::memcpy(event.receipt.binding.digest.data(),
                value->receipt_binding_digest, kDigestBytes);
    std::memcpy(event.receipt.issuer.digest.data(), value->receipt_issuer_digest,
                kDigestBytes);
    event.receipt.kind = static_cast<ReceiptKind>(value->receipt_kind);
    event.receipt.outcome =
        static_cast<ReceiptOutcome>(value->receipt_outcome);
    event.receipt.terminal_timestamp_ns =
        value->receipt_terminal_timestamp_ns;
    std::memcpy(event.receipt.nonce.data(), value->receipt_nonce,
                kReceiptNonceBytes);
    std::lock_guard<std::mutex> lock(owner->mutex);
    const auto lifecycle = owner->lifecycles.find(event.binding_digest);
    if (lifecycle == owner->lifecycles.end()) {
      return ENOENT;
    }
    event.receipt.binding = lifecycle->second.binding;
    const auto producer = owner->producers.find(capsule->producer_id);
    if (producer == owner->producers.end() || !producer->second.has_issuer) {
      return EPERM;
    }
    event.receipt.issuer = producer->second.issuer;
  }
  return submit_event_locked(*owner, std::move(event));
}

int producer_capsule_retire(void *context) noexcept {
  if (context == nullptr) {
    return EINVAL;
  }
  ProducerCapsule *capsule = static_cast<ProducerCapsule *>(context);
  const std::shared_ptr<SharedOwner> owner = capsule->owner;
  if (owner == nullptr) {
    return ESHUTDOWN;
  }
  return submit_producer_retirement_locked(*owner, capsule->producer_id);
}

int producer_capsule_join(void *context, std::uint64_t timeout_ns) noexcept {
  if (context == nullptr || timeout_ns == 0) {
    return EINVAL;
  }
  ProducerCapsule *capsule = static_cast<ProducerCapsule *>(context);
  const std::shared_ptr<SharedOwner> owner = capsule->owner;
  if (owner == nullptr) {
    return ESHUTDOWN;
  }
  try {
    std::unique_lock<std::mutex> lock(owner->mutex);
    const auto completed = [&]() {
      const auto producer = owner->producers.find(capsule->producer_id);
      return producer == owner->producers.end() || producer->second.retired ||
             owner->fatal_code != FatalCode::kNone || owner->closed;
    };
    if (!owner->condition.wait_for(lock, std::chrono::nanoseconds(timeout_ns),
                                   completed)) {
      return ETIMEDOUT;
    }
    const auto producer = owner->producers.find(capsule->producer_id);
    if (producer == owner->producers.end()) {
      return ENOENT;
    }
    return producer->second.retired ? 0 : ESHUTDOWN;
  } catch (...) {
    return EIO;
  }
}

constexpr sglang_terminal_owner_producer_api_v1 kProducerApiV1{
    SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION,
    sizeof(sglang_terminal_owner_producer_api_v1),
    sizeof(sglang_terminal_owner_producer_event_v1),
    SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS,
    &producer_capsule_submit,
    &producer_capsule_retire,
    &producer_capsule_join};

class NativeTerminalOwnerBridge {
public:
  NativeTerminalOwnerBridge(std::size_t input_capacity,
                            std::size_t output_capacity,
                            std::size_t observation_capacity,
                            std::size_t maximum_live_lifecycles,
                            const py::dict &owner_identity,
                            const py::sequence &deadline_table,
                            const py::bytes &deadline_table_digest)
      : owner_(std::make_shared<SharedOwner>()) {
    if (input_capacity == 0 || output_capacity == 0 ||
        observation_capacity == 0 ||
        maximum_live_lifecycles == 0) {
      throw std::invalid_argument("owner queue capacities must be positive");
    }
    owner_->input_capacity = input_capacity;
    owner_->output_capacity = output_capacity;
    owner_->observation_capacity = observation_capacity;
    owner_->maximum_live_lifecycles = maximum_live_lifecycles;
    owner_->owner_identity = process_identity_from_python(owner_identity);
    owner_->deadline_table_digest =
        exact_bytes<kDigestBytes>(deadline_table_digest,
                                  "deadline_table_digest");
    std::array<bool, 11> seen{};
    for (const py::handle item : deadline_table) {
      const py::dict value = py::cast<py::dict>(item);
      DeadlineSpec spec{};
      spec.kind = static_cast<DeadlineKind>(py::cast<int>(value["kind"]));
      const std::uint8_t index = static_cast<std::uint8_t>(spec.kind);
      if (index == 0 || index >= owner_->deadline_specs.size() || seen[index]) {
        throw std::invalid_argument("deadline table kinds must be complete and unique");
      }
      spec.duration_ns = py::cast<std::uint64_t>(value["duration_ns"]);
      spec.process_fatal = py::cast<bool>(value["process_fatal"]);
      spec.starts_at = py::cast<std::string>(value["starts_at"]);
      spec.timeout_outcome =
          py::cast<std::string>(value["timeout_outcome"]);
      if (spec.duration_ns == 0 || spec.starts_at.empty() ||
          spec.timeout_outcome.empty()) {
        throw std::invalid_argument("deadline table contains an empty field");
      }
      owner_->deadline_specs[index] = std::move(spec);
      seen[index] = true;
    }
    for (std::uint8_t index = 1; index < seen.size(); ++index) {
      if (!seen[index]) {
        throw std::invalid_argument("deadline table is incomplete");
      }
    }
    owner_->input_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    owner_->output_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    owner_->observation_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    owner_->timer_fd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    owner_->shutdown_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (owner_->input_fd < 0 || owner_->output_fd < 0 ||
        owner_->observation_fd < 0 ||
        owner_->timer_fd < 0 || owner_->shutdown_fd < 0) {
      const int error = errno;
      close_fd(owner_->input_fd);
      close_fd(owner_->output_fd);
      close_fd(owner_->observation_fd);
      close_fd(owner_->timer_fd);
      close_fd(owner_->shutdown_fd);
      throw std::system_error(error, std::generic_category(),
                              "native terminal owner fd creation failed");
    }
  }

  ~NativeTerminalOwnerBridge() { abort_and_close(); }

  NativeTerminalOwnerBridge(const NativeTerminalOwnerBridge &) = delete;
  NativeTerminalOwnerBridge &
  operator=(const NativeTerminalOwnerBridge &) = delete;

  void enable_forward_independent_handoff() {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->started || owner_->closed) {
      throw std::runtime_error(
          "terminal handoff must be enabled before owner startup");
    }
    if (owner_->handoff_enabled) {
      throw std::runtime_error("terminal handoff cannot be enabled twice");
    }
    if (PyGILState_Check() == 0) {
      throw std::runtime_error(
          "terminal handoff activation requires the interpreter GIL");
    }
    owner_->handoff_enabled = true;
  }

  void start() {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->started || owner_->closed) {
      throw std::runtime_error("native terminal owner cannot restart");
    }
    owner_->started = true;
    owner_->reactor = std::thread(reactor_main, owner_);
    if (!owner_->input_queue.empty()) {
      signal_fd_locked(*owner_, owner_->input_fd);
    }
  }

  void register_producer(std::uint64_t producer_id, const std::string &name,
                         int producer_class, int allowed_role,
                         const py::object &issuer) {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->started || owner_->closed) {
      throw std::runtime_error("producers must register before owner start");
    }
    ProducerRegistration registration{};
    registration.producer_id = producer_id;
    registration.name = name;
    registration.producer_class = static_cast<ProducerClass>(producer_class);
    registration.allowed_role = static_cast<OwnerRole>(allowed_role);
    if (!issuer.is_none()) {
      registration.has_issuer = true;
      registration.issuer =
          process_identity_from_python(py::cast<py::dict>(issuer));
    }
    if (registration.name.empty()) {
      throw std::invalid_argument("producer name must be non-empty");
    }
    if (registration.producer_class != ProducerClass::kLocal &&
        registration.producer_class != ProducerClass::kReceipt &&
        registration.producer_class != ProducerClass::kControl &&
        registration.producer_class != ProducerClass::kQualification) {
      throw std::invalid_argument("producer class is invalid");
    }
    if (registration.allowed_role != OwnerRole::kSource &&
        registration.allowed_role != OwnerRole::kDecode) {
      throw std::invalid_argument("producer role is invalid");
    }
    if (registration.producer_class == ProducerClass::kLocal &&
        registration.has_issuer) {
      throw std::invalid_argument("local producer cannot claim receipt authority");
    }
    if (registration.producer_class != ProducerClass::kLocal &&
        !registration.has_issuer) {
      throw std::invalid_argument("authenticated producer requires an issuer");
    }
    if (!owner_->producers.emplace(producer_id, registration).second) {
      throw std::invalid_argument("producer ID was already registered");
    }
  }

  int register_source(const py::dict &registration) {
    RequestBinding binding =
        binding_from_python(py::cast<py::dict>(registration["binding"]));
    PublicationIdentity publication = publication_from_python(
        py::cast<py::dict>(registration["publication_identity"]));
    if (binding.owner.role != OwnerRole::kSource ||
        publication.room_id != binding.room_id ||
        publication.request_generation != binding.request_generation) {
      throw std::invalid_argument("source registration identities disagree");
    }
    Lifecycle lifecycle{};
    lifecycle.binding = binding;
    lifecycle.publication_identity = publication;
    lifecycle.role = OwnerRole::kSource;
    lifecycle.phase = static_cast<std::uint8_t>(SourcePhase::kFrozen);
    lifecycle.live_resources = kSourceResourceMask;
    add_trusted_issuers(registration, lifecycle);
    lifecycle.trusted_issuers.insert(owner_->owner_identity.digest);
    py::gil_scoped_release release;
    return submit_lifecycle_locked(*owner_, std::move(lifecycle));
  }

  int register_decode(const py::dict &registration) {
    RequestBinding binding =
        binding_from_python(py::cast<py::dict>(registration["binding"]));
    if (binding.owner.role != OwnerRole::kDecode ||
        !registration["publication_identity"].is_none()) {
      throw std::invalid_argument("decode registration identities disagree");
    }
    Lifecycle lifecycle{};
    lifecycle.binding = binding;
    lifecycle.role = OwnerRole::kDecode;
    lifecycle.phase = static_cast<std::uint8_t>(DecodePhase::kPrepared);
    lifecycle.live_resources = kDecodeResourceMask;
    add_trusted_issuers(registration, lifecycle);
    lifecycle.trusted_issuers.insert(owner_->owner_identity.digest);
    py::gil_scoped_release release;
    return submit_lifecycle_locked(*owner_, std::move(lifecycle));
  }

  int submit_event(const py::dict &value) {
    Event event = event_from_python(value);
    py::gil_scoped_release release;
    return submit_event_locked(*owner_, std::move(event));
  }

  py::capsule producer_api() const {
    return py::capsule(
        const_cast<sglang_terminal_owner_producer_api_v1 *>(&kProducerApiV1),
        SGLANG_TERMINAL_OWNER_PRODUCER_API_CAPSULE_NAME);
  }

  py::capsule producer_capsule(std::uint64_t producer_id) const {
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      if (owner_->producers.count(producer_id) != 1) {
        throw std::invalid_argument("producer capsule requires registration");
      }
    }
    auto *capsule = new ProducerCapsule{owner_, producer_id};
    return py::capsule(
        capsule, SGLANG_TERMINAL_OWNER_PRODUCER_CONTEXT_CAPSULE_NAME,
        [](PyObject *value) {
          void *pointer = PyCapsule_GetPointer(
              value, SGLANG_TERMINAL_OWNER_PRODUCER_CONTEXT_CAPSULE_NAME);
          delete static_cast<ProducerCapsule *>(pointer);
        });
  }

  int output_fileno() const {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    return owner_->output_fd;
  }

  int observation_fileno() const {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    return owner_->observation_fd;
  }

  py::list drain_outputs() {
    std::deque<Output> outputs;
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      if (owner_->output_drain_active) {
        throw std::runtime_error("native output drain is single-consumer");
      }
      owner_->output_drain_active = true;
      const int error = consume_fd(owner_->output_fd);
      if (error != 0) {
        owner_->output_drain_active = false;
        owner_->fatal_system_error = error;
        quarantine_all_locked(*owner_, FatalCode::kEventfdFailure, nullptr,
                              "output eventfd read failed");
        throw std::system_error(error, std::generic_category(),
                                "output eventfd read failed");
      }
      outputs.swap(owner_->output_queue);
      outputs.insert(outputs.end(),
                     std::make_move_iterator(owner_->fatal_output_queue.begin()),
                     std::make_move_iterator(owner_->fatal_output_queue.end()));
      owner_->fatal_output_queue.clear();
      for (const Output &output : outputs) {
        for (const Action &action : output.actions) {
          if (!owner_->output_drain_action_ids.insert(action.action_id).second) {
            owner_->output_drain_active = false;
            quarantine_all_locked(*owner_, FatalCode::kInternalError, nullptr,
                                  "output drain duplicated an action identity");
            throw std::runtime_error(
                "output drain duplicated an action identity");
          }
        }
      }
      owner_->output_drain_active =
          !owner_->output_drain_action_ids.empty();
      owner_->condition.notify_all();
    }
    py::list result;
    for (const Output &output : outputs) {
      result.append(output_to_python(output));
    }
    return result;
  }

  py::list drain_observations() {
    std::deque<Observation> observations;
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      const int error = consume_fd(owner_->observation_fd);
      owner_->observation_wake_armed = false;
      if (error != 0) {
        ++owner_->observation_eventfd_error_count;
        owner_->dropped_observation_count += owner_->observation_queue.size();
        owner_->observation_queue.clear();
        owner_->condition.notify_all();
        return py::list();
      }
      observations.swap(owner_->observation_queue);
      owner_->delivered_observation_count += observations.size();
      owner_->condition.notify_all();
    }
    py::list result;
    for (const Observation &observation : observations) {
      result.append(observation_to_python(observation));
    }
    return result;
  }

  void acknowledge_action(std::uint64_t action_id) {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    const auto action = owner_->pending_actions.find(action_id);
    if (action == owner_->pending_actions.end()) {
      const FatalCode code = owner_->consumed_actions.count(action_id) == 1
                                 ? FatalCode::kActionReplay
                                 : FatalCode::kUnknownAction;
      quarantine_all_locked(*owner_, code, nullptr,
                            "action acknowledgement was absent or replayed");
      throw std::runtime_error("action acknowledgement was absent or replayed");
    }
    owner_->pending_actions.erase(action);
    owner_->consumed_actions.insert(action_id);
    if (owner_->output_drain_active) {
      if (owner_->output_drain_action_ids.erase(action_id) != 1) {
        quarantine_all_locked(*owner_, FatalCode::kUnknownAction, nullptr,
                              "action acknowledgement bypassed output drain");
        throw std::runtime_error(
            "action acknowledgement bypassed output drain");
      }
      if (owner_->output_drain_action_ids.empty()) {
        owner_->output_drain_active = false;
      }
    }
    owner_->condition.notify_all();
  }

  void complete_forward_independent_handoff(std::uint64_t action_id) {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (!owner_->handoff_enabled) {
      throw std::runtime_error("terminal handoff is not enabled");
    }
    const auto pending = owner_->handoff_action_ids.find(action_id);
    if (pending == owner_->handoff_action_ids.end()) {
      enter_handoff_fatal_locked(*owner_, FatalCode::kHandoffAuthority,
                                 "terminal handoff action completion was "
                                 "unknown, excluded, or replayed");
      throw std::runtime_error(
          "terminal handoff action completion was absent or replayed");
    }
    owner_->handoff_action_ids.erase(pending);
    ++owner_->handoff_completion_count;
    owner_->condition.notify_all();
  }

  void fail_action_delivery(std::uint64_t action_id,
                            const std::string &reason) {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    const auto action = owner_->pending_actions.find(action_id);
    if (action == owner_->pending_actions.end()) {
      throw std::runtime_error(
          "failed action delivery was absent or already resolved");
    }
    Event trigger{};
    trigger.binding_digest = action->second.binding.digest;
    trigger.enqueued_ns = owner_now_ns_locked(*owner_);
    trigger.kind = action->second.binding.owner.role == OwnerRole::kSource
                       ? EventKind::kSourceInboxOverflow
                       : EventKind::kDecodeInboxOverflow;
    fail_forward_independent_handoff_locked(*owner_, action_id);
    owner_->pending_actions.erase(action);
    owner_->output_drain_action_ids.erase(action_id);
    if (owner_->output_drain_action_ids.empty()) {
      owner_->output_drain_active = false;
    }
    if (owner_->fatal_code == FatalCode::kNone) {
      quarantine_all_locked(*owner_, FatalCode::kOutputQueueOverflow,
                            &trigger, reason);
    }
    owner_->condition.notify_all();
  }

  py::dict inventory() const {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    return inventory_locked();
  }

  bool wait_for_lifecycle_registration(const py::bytes &binding_digest,
                                       double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("registration wait must be positive");
    }
    const Digest digest =
        exact_bytes<kDigestBytes>(binding_digest, "binding digest");
    py::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(owner_->mutex);
    const bool reached = owner_->condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [&]() {
          return owner_->lifecycles.count(digest) == 1 ||
                 owner_->fatal_code != FatalCode::kNone || owner_->closed;
        });
    return reached && owner_->lifecycles.count(digest) == 1;
  }

  bool wait_for_forward_independent_handoff(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("handoff wait must be positive");
    }
    py::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(owner_->mutex);
    const bool reached = owner_->condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [&]() {
          return owner_->handoff_callback_active ||
                 owner_->fatal_code != FatalCode::kNone || owner_->closed;
        });
    return reached && owner_->handoff_callback_active;
  }

#ifdef SGLANG_TERMINAL_OWNER_TESTING
  bool wait_for_process_fatal(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("process-fatal wait must be positive");
    }
    py::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(owner_->mutex);
    return owner_->condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [&]() {
          return owner_->fatal_code != FatalCode::kNone || owner_->closed;
        }) && owner_->fatal_code != FatalCode::kNone;
  }

  py::dict lifecycle_snapshot(const py::bytes &binding_digest) const {
    const Digest digest =
        exact_bytes<kDigestBytes>(binding_digest, "binding digest");
    std::lock_guard<std::mutex> lock(owner_->mutex);
    const auto entry = owner_->lifecycles.find(digest);
    if (entry == owner_->lifecycles.end()) {
      throw std::invalid_argument("lifecycle snapshot binding is unknown");
    }
    const Lifecycle &lifecycle = entry->second;
    py::dict result;
    result["binding"] = binding_to_python(lifecycle.binding);
    result["role"] = static_cast<int>(lifecycle.role);
    result["phase"] = lifecycle.phase;
    result["live_resources"] = lifecycle.live_resources;
    result["retired_resources"] = lifecycle.retired_resources;
    result["quarantined_resources"] = lifecycle.quarantined_resources;
    result["armed_deadline_mask"] = lifecycle.armed_deadline_mask;
    result["process_fatal"] = owner_->fatal_code != FatalCode::kNone;
    return result;
  }

  void enable_test_clock(std::uint64_t now_ns) {
    if (now_ns == 0) {
      throw std::invalid_argument("test clock must be positive");
    }
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->started || owner_->closed || owner_->test_clock_enabled) {
      throw std::runtime_error(
          "test clock must be enabled once before owner startup");
    }
    owner_->test_clock_enabled = true;
    owner_->test_now_ns = now_ns;
    owner_->condition.notify_all();
  }

  void set_test_clock(std::uint64_t now_ns) {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (!owner_->test_clock_enabled || !owner_->started || owner_->closed ||
        now_ns < owner_->test_now_ns) {
      throw std::runtime_error(
          "test clock requires a running owner and monotonic time");
    }
    owner_->test_now_ns = now_ns;
    owner_->condition.notify_all();
  }

  void expire_deadlines_for_test() {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (!owner_->test_clock_enabled || !owner_->started || owner_->closed) {
      throw std::runtime_error(
          "deadline expiry requires a running deterministic test clock");
    }
    expire_deadlines_locked(*owner_);
  }

  void abort_active_qualification_for_test() {
    std::thread reactor;
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      if (!owner_->started || !owner_->qualification.running ||
          owner_->closed) {
        throw std::runtime_error(
            "qualification abort requires an active test population");
      }
      owner_->admission_open = false;
      owner_->event_admission_open = false;
      owner_->qualification.running = false;
      owner_->stop_requested = true;
      for (auto &entry : owner_->producers) {
        entry.second.retirement_requested = true;
        entry.second.retired = true;
      }
      owner_->producers_joined = true;
      owner_->input_queue.clear();
      for (auto &entry : owner_->lifecycles) {
        quarantine_live(entry.second);
      }
      signal_fd_locked(*owner_, owner_->shutdown_fd);
      reactor = std::move(owner_->reactor);
    }
    if (reactor.joinable()) {
      reactor.join();
    }
    std::lock_guard<std::mutex> lock(owner_->mutex);
    close_fds_locked();
    owner_->closed = true;
    owner_->condition.notify_all();
  }
#endif

  void start_qualification(std::size_t machine_count,
                           double minimum_duration_seconds,
                           std::uint64_t minimum_transition_count) {
    if (machine_count == 0 || minimum_duration_seconds <= 0.0 ||
        minimum_transition_count == 0) {
      throw std::invalid_argument("qualification bounds must be positive");
    }
    if (machine_count > kQualificationMaximumMachineCount) {
      throw std::invalid_argument(
          "qualification machine count exceeds its evidence bound");
    }
    if (minimum_transition_count >
        kQualificationMaximumFullPathSampleCount *
            kQualificationMeasuredHopCount) {
      throw std::invalid_argument(
          "qualification transition floor exceeds its evidence bound");
    }
    if (machine_count > owner_->input_capacity) {
      throw std::invalid_argument(
          "qualification bootstrap exceeds the native input capacity");
    }
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (!owner_->started || owner_->qualification.running ||
        owner_->qualification.complete) {
      throw std::runtime_error("qualification lifecycle is invalid");
    }
    if (!owner_->producers.empty() || !owner_->lifecycles.empty() ||
        owner_->next_owner_sequence != 0) {
      throw std::runtime_error(
          "qualification requires a dedicated unused native owner");
    }
    QualificationState &qualification = owner_->qualification;
    qualification.running = true;
    qualification.machine_count = machine_count;
    qualification.minimum_duration_ns = static_cast<std::uint64_t>(
        minimum_duration_seconds * 1'000'000'000.0);
    qualification.minimum_transition_count = minimum_transition_count;
    qualification.started_ns = owner_now_ns_locked(*owner_);
    qualification.owner_sequence_start = owner_->next_owner_sequence;
    qualification.bindings.resize(machine_count);
    qualification.producer_ids.resize(machine_count);
    qualification.producer_sequences.assign(machine_count, 0);
    qualification.generations.assign(machine_count, 0);
    qualification.completed_generations.assign(machine_count, 0);
    qualification.hops.assign(machine_count, 0);
    qualification.path_latency_accumulators.assign(machine_count, 0);
    qualification.first_audit_traces.resize(machine_count);
    qualification.last_audit_traces.resize(machine_count);
    for (std::size_t index = 0; index < machine_count; ++index) {
      const std::uint64_t producer_id = 0xf000000000000000ULL + index;
      ProducerRegistration producer{};
      producer.producer_id = producer_id;
      producer.name = "qualification-" + std::to_string(index);
      producer.producer_class = ProducerClass::kQualification;
      producer.allowed_role = OwnerRole::kSource;
      producer.has_issuer = true;
      producer.issuer = owner_->owner_identity;
      if (!owner_->producers.emplace(producer_id, producer).second) {
        throw std::runtime_error("qualification producer ID collision");
      }
      RequestBinding binding{};
      binding.room_id = 0xf000000000000000ULL + index;
      binding.owner = owner_->owner_identity;
      for (std::size_t byte = 0; byte < 8; ++byte) {
        binding.digest[byte] =
            static_cast<std::uint8_t>(index >> (byte * 8));
      }
      binding.digest[31] = static_cast<std::uint8_t>(index);
      Lifecycle lifecycle{};
      lifecycle.binding = binding;
      lifecycle.role = OwnerRole::kSource;
      lifecycle.phase = static_cast<std::uint8_t>(SourcePhase::kFrozen);
      lifecycle.live_resources = kSourceResourceMask;
      lifecycle.trusted_issuers.insert(owner_->owner_identity.digest);
      InputCommand command{};
      command.kind = InputKind::kRegisterLifecycle;
      command.lifecycle = std::move(lifecycle);
      owner_->input_queue.push_back(std::move(command));
      qualification.bindings[index] = binding.digest;
      qualification.producer_ids[index] = producer_id;
    }
    signal_fd_locked(*owner_, owner_->input_fd);
  }

  bool qualification_join(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("qualification timeout must be positive");
    }
    std::unique_lock<std::mutex> lock(owner_->mutex);
    const bool reached = owner_->qualification_condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [this]() {
          return owner_->qualification.complete ||
                 owner_->fatal_code != FatalCode::kNone;
        });
    if (owner_->fatal_code != FatalCode::kNone) {
      throw std::runtime_error(std::string("native owner fatal: ") +
                               fatal_name(owner_->fatal_code));
    }
    return reached && owner_->qualification.complete;
  }

  py::dict qualification_summary() const {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    const QualificationState &qualification = owner_->qualification;
    if (!qualification.complete || !qualification.summary_complete) {
      throw std::runtime_error("qualification summary is not complete");
    }
    py::dict result;
    result["machine_count"] = qualification.machine_count;
    result["measured_hop_count"] = kQualificationMeasuredHopCount;
    result["lifecycle_hop_count"] = kQualificationLifecycleHopCount;
    result["statistics_sample_capacity"] =
        kQualificationMaximumFullPathSampleCount;
    result["minimum_duration_ns"] = qualification.minimum_duration_ns;
    result["minimum_transition_count"] =
        qualification.minimum_transition_count;
    result["started_ns"] = qualification.started_ns;
    result["ended_ns"] = qualification.ended_ns;
    result["transition_count"] = qualification.transition_count;
    result["lifecycle_transition_count"] =
        qualification.lifecycle_transition_count;
    result["sample_count"] = qualification.sample_count;
    result["owner_sequence_start"] = qualification.owner_sequence_start;
    result["owner_sequence_end"] = qualification.owner_sequence_end;
    result["raw_trace_retained_count"] = 0;

    py::list transition_classes;
    for (std::size_t hop = 0; hop < kQualificationMeasuredHopCount; ++hop) {
      py::dict value =
          latency_statistics_to_python(qualification.hop_statistics.at(hop));
      value["hop_index"] = hop;
      value["event_kind"] =
          static_cast<int>(qualification_event_kind(hop));
      transition_classes.append(std::move(value));
    }
    result["transition_classes"] = std::move(transition_classes);
    result["seven_hop_path"] =
        latency_statistics_to_python(qualification.path_statistics);
    result["completed_generations_by_machine"] =
        qualification.completed_generations;
    result["producer_sequences_by_machine"] =
        qualification.producer_sequences;

    py::list first_audit;
    py::list last_audit;
    std::size_t audit_sample_count = 0;
    for (std::size_t machine = 0; machine < qualification.machine_count;
         ++machine) {
      for (std::size_t hop = 0; hop < kQualificationMeasuredHopCount; ++hop) {
        const std::optional<Trace> &first =
            qualification.first_audit_traces.at(machine).at(hop);
        const std::optional<Trace> &last =
            qualification.last_audit_traces.at(machine).at(hop);
        if (first.has_value()) {
          first_audit.append(trace_to_python(first.value()));
          ++audit_sample_count;
        }
        if (last.has_value()) {
          last_audit.append(trace_to_python(last.value()));
          ++audit_sample_count;
        }
      }
    }
    result["audit_sample_bound"] = qualification.machine_count *
                                    kQualificationMeasuredHopCount * 2;
    result["audit_sample_count"] = audit_sample_count;
    result["first_audit_samples"] = std::move(first_audit);
    result["last_audit_samples"] = std::move(last_audit);
    return result;
  }

  void stop_admission() {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    owner_->admission_open = false;
  }

  bool join_producers() {
    std::unique_lock<std::mutex> lock(owner_->mutex);
    if (owner_->closed) {
      return false;
    }
    if (owner_->abort_started) {
      owner_->producers_joined = std::all_of(
          owner_->producers.begin(), owner_->producers.end(),
          [](const auto &entry) { return entry.second.retired; });
      return owner_->producers_joined;
    }
    const bool every_retirement_requested = std::all_of(
        owner_->producers.begin(), owner_->producers.end(),
        [](const auto &entry) {
          return entry.second.retirement_requested || entry.second.retired;
        });
    if (!every_retirement_requested) {
      return false;
    }
    if (!owner_->producer_join_requested) {
      owner_->producer_join_requested = true;
      owner_->event_admission_open = false;
      signal_fd_locked(*owner_, owner_->input_fd);
    }
    owner_->condition.wait(lock, [&]() {
      return owner_->producer_join_barrier_complete ||
             owner_->closed;
    });
    return owner_->producers_joined;
  }

  int retire_python_producer(std::uint64_t producer_id) {
    return submit_producer_retirement_locked(*owner_, producer_id);
  }

  bool wait_for_producer_retirement(std::uint64_t producer_id,
                                    double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("producer retirement wait must be positive");
    }
    std::unique_lock<std::mutex> lock(owner_->mutex);
    if (owner_->producers.count(producer_id) != 1) {
      throw std::invalid_argument("producer retirement requires registration");
    }
    return owner_->condition.wait_for(
               lock, std::chrono::duration<double>(timeout_seconds), [&]() {
                 return owner_->producers.at(producer_id).retired ||
                        owner_->fatal_code != FatalCode::kNone ||
                        owner_->closed;
               }) &&
           owner_->producers.at(producer_id).retired;
  }

  bool wait_for_output_projection(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument(
          "output projection wait must be positive");
    }
    std::unique_lock<std::mutex> lock(owner_->mutex);
    return owner_->condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [&]() {
          return owner_->input_queue.empty() &&
                 owner_->output_queue.empty() &&
                 owner_->fatal_output_queue.empty() &&
                 owner_->pending_actions.empty() &&
                 !owner_->output_drain_active &&
                 !owner_->qualification.running;
        });
  }

  bool wait_for_output_quiescence(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument(
          "output quiescence wait must be positive");
    }
    std::unique_lock<std::mutex> lock(owner_->mutex);
    if (!owner_->producers_joined) {
      return false;
    }
    return owner_->condition.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [&]() {
          return owner_->input_queue.empty() &&
                 owner_->output_queue.empty() &&
                 owner_->fatal_output_queue.empty() &&
                 owner_->pending_actions.empty() &&
                 !owner_->output_drain_active &&
                 !owner_->qualification.running;
        });
  }

  void close() {
    std::thread reactor;
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      if (owner_->closed) {
        return;
      }
      owner_->admission_open = false;
      owner_->event_admission_open = false;
      bool retained = !owner_->input_queue.empty() ||
                      !owner_->output_queue.empty() ||
                      !owner_->fatal_output_queue.empty() ||
                      !owner_->pending_actions.empty() ||
                      owner_->output_drain_active ||
                      !owner_->output_drain_action_ids.empty() ||
                      !owner_->handoff_action_ids.empty() ||
                      !owner_->producers_joined;
      for (const auto &entry : owner_->lifecycles) {
        retained = retained || entry.second.live_resources != 0;
      }
      if (retained) {
        quarantine_all_locked(*owner_, FatalCode::kCloseWithRetainedInventory,
                              nullptr,
                              "owner close retained unresolved inventory");
        throw std::runtime_error("owner close retained unresolved inventory");
      }
      owner_->stop_requested = true;
      owner_->qualification.running = false;
      signal_fd_locked(*owner_, owner_->shutdown_fd);
      reactor = std::move(owner_->reactor);
    }
    if (reactor.joinable()) {
      reactor.join();
    }
    std::lock_guard<std::mutex> lock(owner_->mutex);
    close_fds_locked();
    owner_->closed = true;
    owner_->condition.notify_all();
  }

  void begin_abort() {
    std::thread reactor;
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      if (owner_->closed) {
        return;
      }
      if (owner_->abort_started) {
        return;
      }
      owner_->abort_started = true;
      owner_->admission_open = false;
      owner_->event_admission_open = false;
      owner_->stop_requested = true;
      owner_->qualification.running = false;
      for (InputCommand &command : owner_->input_queue) {
        if (command.kind != InputKind::kRegisterLifecycle) {
          continue;
        }
        register_lifecycle_locked(*owner_, std::move(command.lifecycle));
      }
      for (const InputCommand &command : owner_->input_queue) {
        if (command.kind != InputKind::kRetireProducer) {
          continue;
        }
        auto producer = owner_->producers.find(command.producer_id);
        if (producer != owner_->producers.end()) {
          producer->second.retired = true;
        }
      }
      owner_->input_queue.clear();
      if (owner_->fatal_code == FatalCode::kNone) {
        quarantine_all_locked(*owner_, FatalCode::kDependencyDeath, nullptr,
                              "owner aborted before clean close");
      } else {
        publish_quarantine_for_live_locked(*owner_, owner_->fatal_code,
                                           nullptr);
      }
      signal_fd_locked(*owner_, owner_->shutdown_fd);
      reactor = std::move(owner_->reactor);
    }
    if (reactor.joinable()) {
      reactor.join();
    }
    {
      std::lock_guard<std::mutex> lock(owner_->mutex);
      owner_->reactor_stopped = true;
      owner_->condition.notify_all();
    }
  }

  void close_aborted() {
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->closed) {
      return;
    }
    if (!owner_->abort_started || !owner_->reactor_stopped ||
        owner_->fatal_code == FatalCode::kNone || !owner_->producers_joined) {
      throw std::runtime_error(
          "aborted close requires terminal authority and producer join");
    }
    if (!owner_->output_queue.empty() || !owner_->pending_actions.empty() ||
        !owner_->fatal_output_queue.empty() ||
        owner_->output_drain_active ||
        !owner_->output_drain_action_ids.empty() ||
        !owner_->handoff_action_ids.empty()) {
      throw std::runtime_error(
          "aborted close retained unrouted terminal authority");
    }
    close_fds_locked();
    owner_->closed = true;
  }

  void abort_and_close() noexcept {
    if (owner_ == nullptr) {
      return;
    }
    try {
      begin_abort();
    } catch (...) {
    }
    std::lock_guard<std::mutex> lock(owner_->mutex);
    if (owner_->closed) {
      return;
    }
    owner_->output_queue.clear();
    owner_->fatal_output_queue.clear();
    owner_->pending_actions.clear();
    owner_->output_drain_action_ids.clear();
    owner_->handoff_action_ids.clear();
    owner_->output_drain_active = false;
    close_fds_locked();
    owner_->closed = true;
    owner_->condition.notify_all();
  }

private:
  static void add_trusted_issuers(const py::dict &registration,
                                  Lifecycle &lifecycle) {
    for (const py::handle item : py::cast<py::sequence>(
             registration["trusted_issuers"])) {
      const ProcessIdentity issuer =
          process_identity_from_python(py::cast<py::dict>(item));
      if (!lifecycle.trusted_issuers.insert(issuer.digest).second) {
        throw std::invalid_argument("trusted issuer digest was duplicated");
      }
    }
  }

  py::dict inventory_locked() const {
    py::dict result;
    std::size_t active_source_count = 0;
    std::size_t active_decode_count = 0;
    std::size_t retired_count = 0;
    std::size_t quarantined_count = 0;
    std::size_t armed_deadline_count = 0;
    py::list bindings;
    py::list quarantined_binding_digests;
    for (const auto &entry : owner_->lifecycles) {
      const Lifecycle &lifecycle = entry.second;
      if (lifecycle.live_resources != 0) {
        if (lifecycle.role == OwnerRole::kSource) {
          ++active_source_count;
        } else {
          ++active_decode_count;
        }
      }
      if (lifecycle.live_resources == 0 &&
          lifecycle.quarantined_resources == 0) {
        ++retired_count;
      }
      if (lifecycle.quarantined_resources != 0) {
        ++quarantined_count;
        quarantined_binding_digests.append(to_bytes(lifecycle.binding.digest));
      }
      armed_deadline_count += static_cast<std::size_t>(
          __builtin_popcount(static_cast<unsigned>(
              lifecycle.armed_deadline_mask)));
      py::dict value;
      value["binding"] = binding_to_python(lifecycle.binding);
      value["role"] = static_cast<int>(lifecycle.role);
      value["phase"] = lifecycle.phase;
      value["live_resources"] = lifecycle.live_resources;
      value["retired_resources"] = lifecycle.retired_resources;
      value["quarantined_resources"] = lifecycle.quarantined_resources;
      value["armed_deadline_mask"] = lifecycle.armed_deadline_mask;
      bindings.append(std::move(value));
    }
    py::list deadline_table;
    for (std::uint8_t index = 1; index < owner_->deadline_specs.size();
         ++index) {
      const DeadlineSpec &spec = owner_->deadline_specs[index];
      py::dict value;
      value["kind"] = static_cast<int>(spec.kind);
      value["duration_ns"] = spec.duration_ns;
      value["process_fatal"] = spec.process_fatal;
      value["starts_at"] = spec.starts_at;
      value["timeout_outcome"] = spec.timeout_outcome;
      deadline_table.append(std::move(value));
    }
    result["owner_identity"] = process_identity_to_python(owner_->owner_identity);
    result["deadline_table_digest"] = to_bytes(owner_->deadline_table_digest);
    result["deadline_table"] = std::move(deadline_table);
    result["input_capacity"] = owner_->input_capacity;
    result["output_capacity"] = owner_->output_capacity;
    result["fatal_output_capacity"] = owner_->maximum_live_lifecycles;
    result["observation_capacity"] = owner_->observation_capacity;
    result["queued_input_count"] = owner_->input_queue.size();
    result["queued_output_count"] =
        owner_->output_queue.size() + owner_->fatal_output_queue.size();
    result["queued_fatal_output_count"] = owner_->fatal_output_queue.size();
    result["queued_observation_count"] = owner_->observation_queue.size();
    result["delivered_observation_count"] =
        owner_->delivered_observation_count;
    result["dropped_observation_count"] = owner_->dropped_observation_count;
    result["observation_count"] = owner_->observation_count;
    result["observation_eventfd_error_count"] =
        owner_->observation_eventfd_error_count;
    result["registered_producer_count"] = owner_->producers.size();
    result["joined_producer_count"] = std::count_if(
        owner_->producers.begin(), owner_->producers.end(),
        [](const auto &entry) { return entry.second.retired; });
    result["active_source_count"] = active_source_count;
    result["active_decode_count"] = active_decode_count;
    result["safely_retired_count"] = retired_count;
    result["quarantined_count"] = quarantined_count;
    result["quarantined_binding_digests"] =
        std::move(quarantined_binding_digests);
    result["armed_deadline_count"] = armed_deadline_count;
    result["transition_count"] = owner_->next_owner_sequence;
    result["action_count"] = owner_->total_action_count;
    result["qualification_trace_count"] = 0;
    result["draining"] = !owner_->admission_open && !owner_->closed;
    result["input_eventfd_open"] = owner_->input_fd >= 0;
    result["output_eventfd_open"] = owner_->output_fd >= 0;
    result["observation_eventfd_open"] = owner_->observation_fd >= 0;
    if (owner_->fatal_code == FatalCode::kNone) {
      result["fatal_binding_digest"] = py::none();
    } else {
      result["fatal_binding_digest"] = to_bytes(owner_->fatal_binding);
    }
    result["lifecycles"] = std::move(bindings);
    result["input_queue_count"] = owner_->input_queue.size();
    result["output_queue_count"] = owner_->output_queue.size();
    result["fatal_output_queue_count"] = owner_->fatal_output_queue.size();
    result["pending_action_count"] = owner_->pending_actions.size();
    result["pending_handoff_action_count"] =
        owner_->handoff_action_ids.size();
    result["completed_handoff_action_count"] =
        owner_->handoff_completion_count;
    result["handoff_callback_count"] = owner_->handoff_callback_count;
    result["handoff_enabled"] = owner_->handoff_enabled;
    result["handoff_callback_scheduled"] =
        owner_->handoff_callback_scheduled;
    result["handoff_callback_active"] = owner_->handoff_callback_active;
    result["scheduled_handoff_watermark"] =
        owner_->scheduled_handoff_watermark;
    result["active_handoff_watermark"] = owner_->active_handoff_watermark;
    result["output_drain_active"] = owner_->output_drain_active;
    result["producer_count"] = owner_->producers.size();
    py::list producers;
    for (const auto &entry : owner_->producers) {
      const ProducerRegistration &producer = entry.second;
      py::dict value;
      value["producer_id"] = producer.producer_id;
      value["name"] = producer.name;
      value["producer_class"] = static_cast<int>(producer.producer_class);
      value["allowed_role"] = static_cast<int>(producer.allowed_role);
      if (producer.has_issuer) {
        value["authenticated_issuer"] =
            process_identity_to_python(producer.issuer);
      } else {
        value["authenticated_issuer"] = py::none();
      }
      value["next_sequence"] = producer.next_dispatch_sequence;
      value["next_submission_sequence"] = producer.next_submission_sequence;
      value["retirement_requested"] = producer.retirement_requested;
      value["retired"] = producer.retired;
      producers.append(std::move(value));
    }
    result["producers"] = std::move(producers);
    result["owner_transition_count"] = owner_->next_owner_sequence;
    result["admission_open"] = owner_->admission_open;
    result["event_admission_open"] = owner_->event_admission_open;
    result["producers_joined"] = owner_->producers_joined;
    result["started"] = owner_->started;
    result["closed"] = owner_->closed;
    result["fatal_code"] = static_cast<int>(owner_->fatal_code);
    result["fatal_name"] = fatal_name(owner_->fatal_code);
    result["fatal_system_error"] = owner_->fatal_system_error;
    result["fatal_producer_id"] = owner_->fatal_producer_id;
    result["fatal_producer_sequence"] = owner_->fatal_producer_sequence;
    result["fatal_reason"] = owner_->fatal_reason;
    result["fatal_reason_code"] = owner_->fatal_reason_code;
    result["fatal_backend_status"] = owner_->fatal_backend_status;
    result["qualification_running"] = owner_->qualification.running;
    result["qualification_complete"] = owner_->qualification.complete;
    result["qualification_transition_count"] =
        owner_->qualification.transition_count;
    result["qualification_lifecycle_transition_count"] =
        owner_->qualification.lifecycle_transition_count;
    result["qualification_sample_count"] =
        owner_->qualification.sample_count;
    result["qualification_summary_complete"] =
        owner_->qualification.summary_complete;
    result["qualification_started_ns"] = owner_->qualification.started_ns;
    result["qualification_ended_ns"] = owner_->qualification.ended_ns;
    result["qualification_minimum_duration_ns"] =
        owner_->qualification.minimum_duration_ns;
    result["qualification_minimum_transition_count"] =
        owner_->qualification.minimum_transition_count;
    result["qualification_machine_count"] = owner_->qualification.machine_count;
    return result;
  }

  void close_fds_locked() noexcept {
    owner_->dropped_observation_count += owner_->observation_queue.size();
    owner_->observation_queue.clear();
    owner_->observation_wake_armed = false;
    close_fd(owner_->input_fd);
    close_fd(owner_->output_fd);
    close_fd(owner_->observation_fd);
    close_fd(owner_->timer_fd);
    close_fd(owner_->shutdown_fd);
  }

  std::shared_ptr<SharedOwner> owner_;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::dict producer_event_offsets;
  producer_event_offsets["abi_version"] =
      offsetof(sglang_terminal_owner_producer_event_v1, abi_version);
  producer_event_offsets["struct_size"] =
      offsetof(sglang_terminal_owner_producer_event_v1, struct_size);
  producer_event_offsets["binding_digest"] =
      offsetof(sglang_terminal_owner_producer_event_v1, binding_digest);
  producer_event_offsets["event_kind"] =
      offsetof(sglang_terminal_owner_producer_event_v1, event_kind);
  producer_event_offsets["enqueued_ns"] =
      offsetof(sglang_terminal_owner_producer_event_v1, enqueued_ns);
  producer_event_offsets["receipt_binding_digest"] = offsetof(
      sglang_terminal_owner_producer_event_v1, receipt_binding_digest);
  producer_event_offsets["receipt_nonce"] =
      offsetof(sglang_terminal_owner_producer_event_v1, receipt_nonce);
  module.attr("PRODUCER_ABI_VERSION") =
      SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION;
  module.attr("PRODUCER_EVENT_POD_SIZE") =
      sizeof(sglang_terminal_owner_producer_event_v1);
  module.attr("PRODUCER_API_SIZE") =
      sizeof(sglang_terminal_owner_producer_api_v1);
  module.attr("PRODUCER_API_FLAGS") =
      SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS;
  module.attr("PRODUCER_EVENT_OFFSETS") = std::move(producer_event_offsets);
  module.attr("PRODUCER_API_CAPSULE_NAME") =
      SGLANG_TERMINAL_OWNER_PRODUCER_API_CAPSULE_NAME;
  module.attr("PRODUCER_CONTEXT_CAPSULE_NAME") =
      SGLANG_TERMINAL_OWNER_PRODUCER_CONTEXT_CAPSULE_NAME;
  py::class_<NativeTerminalOwnerBridge>(module, "NativeTerminalOwnerBridge")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t,
                    const py::dict &, const py::sequence &,
                    const py::bytes &>(),
           py::arg("input_capacity"), py::arg("output_capacity"),
           py::arg("observation_capacity"),
           py::arg("maximum_live_lifecycles"),
           py::arg("owner_identity"), py::arg("deadline_table"),
           py::arg("deadline_table_digest"))
      .def("enable_forward_independent_handoff",
           &NativeTerminalOwnerBridge::enable_forward_independent_handoff)
      .def("register_producer",
           &NativeTerminalOwnerBridge::register_producer,
           py::arg("producer_id"), py::arg("name"),
           py::arg("producer_class"), py::arg("allowed_role"),
           py::arg("authenticated_issuer") = py::none())
      .def("register_source", &NativeTerminalOwnerBridge::register_source,
           py::arg("registration"))
      .def("register_decode", &NativeTerminalOwnerBridge::register_decode,
           py::arg("registration"))
      .def("start", &NativeTerminalOwnerBridge::start,
           py::call_guard<py::gil_scoped_release>())
      .def("submit_event", &NativeTerminalOwnerBridge::submit_event,
           py::arg("event"))
      .def("producer_api", &NativeTerminalOwnerBridge::producer_api)
      .def("producer_capsule", &NativeTerminalOwnerBridge::producer_capsule,
           py::arg("producer_id"))
      .def("output_fileno", &NativeTerminalOwnerBridge::output_fileno)
      .def("drain_outputs", &NativeTerminalOwnerBridge::drain_outputs)
      .def("observation_fileno",
           &NativeTerminalOwnerBridge::observation_fileno)
      .def("drain_observations",
           &NativeTerminalOwnerBridge::drain_observations)
      .def("acknowledge_action",
           &NativeTerminalOwnerBridge::acknowledge_action,
           py::arg("action_id"), py::call_guard<py::gil_scoped_release>())
      .def("complete_forward_independent_handoff",
           &NativeTerminalOwnerBridge::complete_forward_independent_handoff,
           py::arg("action_id"), py::call_guard<py::gil_scoped_release>())
      .def("fail_action_delivery",
           &NativeTerminalOwnerBridge::fail_action_delivery,
           py::arg("action_id"), py::arg("reason"),
           py::call_guard<py::gil_scoped_release>())
      .def("inventory", &NativeTerminalOwnerBridge::inventory)
      .def("wait_for_lifecycle_registration",
           &NativeTerminalOwnerBridge::wait_for_lifecycle_registration,
           py::arg("binding_digest"), py::arg("timeout_seconds"))
      .def("wait_for_forward_independent_handoff",
           &NativeTerminalOwnerBridge::wait_for_forward_independent_handoff,
           py::arg("timeout_seconds"))
#ifdef SGLANG_TERMINAL_OWNER_TESTING
      .def("wait_for_process_fatal",
           &NativeTerminalOwnerBridge::wait_for_process_fatal,
           py::arg("timeout_seconds"))
      .def("lifecycle_snapshot",
           &NativeTerminalOwnerBridge::lifecycle_snapshot,
           py::arg("binding_digest"))
      .def("enable_test_clock", &NativeTerminalOwnerBridge::enable_test_clock,
           py::arg("now_ns"))
      .def("set_test_clock", &NativeTerminalOwnerBridge::set_test_clock,
           py::arg("now_ns"))
      .def("expire_deadlines_for_test",
           &NativeTerminalOwnerBridge::expire_deadlines_for_test)
      .def("abort_active_qualification_for_test",
           &NativeTerminalOwnerBridge::abort_active_qualification_for_test,
           py::call_guard<py::gil_scoped_release>())
#endif
      .def("start_qualification",
           &NativeTerminalOwnerBridge::start_qualification,
           py::arg("machine_count"), py::arg("minimum_duration_seconds"),
           py::arg("minimum_transition_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("qualification_join",
           &NativeTerminalOwnerBridge::qualification_join,
           py::arg("timeout_seconds"),
           py::call_guard<py::gil_scoped_release>())
      .def("qualification_summary",
           &NativeTerminalOwnerBridge::qualification_summary)
      .def("stop_admission", &NativeTerminalOwnerBridge::stop_admission,
           py::call_guard<py::gil_scoped_release>())
      .def("join_producers", &NativeTerminalOwnerBridge::join_producers,
           py::call_guard<py::gil_scoped_release>())
      .def("retire_python_producer",
           &NativeTerminalOwnerBridge::retire_python_producer,
           py::arg("producer_id"), py::call_guard<py::gil_scoped_release>())
      .def("wait_for_producer_retirement",
           &NativeTerminalOwnerBridge::wait_for_producer_retirement,
           py::arg("producer_id"), py::arg("timeout_seconds"),
           py::call_guard<py::gil_scoped_release>())
      .def("wait_for_output_projection",
           &NativeTerminalOwnerBridge::wait_for_output_projection,
           py::arg("timeout_seconds"),
           py::call_guard<py::gil_scoped_release>())
      .def("wait_for_output_quiescence",
           &NativeTerminalOwnerBridge::wait_for_output_quiescence,
           py::arg("timeout_seconds"),
           py::call_guard<py::gil_scoped_release>())
      .def("begin_abort", &NativeTerminalOwnerBridge::begin_abort,
           py::call_guard<py::gil_scoped_release>())
      .def("close_aborted", &NativeTerminalOwnerBridge::close_aborted,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &NativeTerminalOwnerBridge::close,
           py::call_guard<py::gil_scoped_release>())
      .def("abort_and_close", &NativeTerminalOwnerBridge::abort_and_close,
           py::call_guard<py::gil_scoped_release>());
}
