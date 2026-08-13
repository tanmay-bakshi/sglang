#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kGenerationBytes = 16;

using Generation = std::array<std::uint8_t, kGenerationBytes>;

enum class FatalCode : std::uint32_t {
  kNone = 0,
  kQueueOverflow = 1,
  kEventfdWriteFailure = 2,
  kEventfdReadFailure = 3,
  kDuplicateIdentity = 4,
  kExactGenerationMismatch = 5,
  kSubmissionWithoutArm = 6,
  kInvalidIdentityState = 7,
  kCudaCallbackRegistrationFailure = 8,
  kSubmissionAfterStop = 9,
  kConcurrentDrain = 10,
  kCloseWithActiveCallbacks = 11,
  kCloseWithRetainedInventory = 12,
  kCloseBeforeProducerJoin = 13,
};

const char *fatal_code_name(FatalCode code) noexcept {
  switch (code) {
  case FatalCode::kNone:
    return "none";
  case FatalCode::kQueueOverflow:
    return "queue_overflow";
  case FatalCode::kEventfdWriteFailure:
    return "eventfd_write_failure";
  case FatalCode::kEventfdReadFailure:
    return "eventfd_read_failure";
  case FatalCode::kDuplicateIdentity:
    return "duplicate_identity";
  case FatalCode::kExactGenerationMismatch:
    return "exact_generation_mismatch";
  case FatalCode::kSubmissionWithoutArm:
    return "submission_without_arm";
  case FatalCode::kInvalidIdentityState:
    return "invalid_identity_state";
  case FatalCode::kCudaCallbackRegistrationFailure:
    return "cuda_callback_registration_failure";
  case FatalCode::kSubmissionAfterStop:
    return "submission_after_stop";
  case FatalCode::kConcurrentDrain:
    return "concurrent_drain";
  case FatalCode::kCloseWithActiveCallbacks:
    return "close_with_active_callbacks";
  case FatalCode::kCloseWithRetainedInventory:
    return "close_with_retained_inventory";
  case FatalCode::kCloseBeforeProducerJoin:
    return "close_before_producer_join";
  }
  return "unknown";
}

struct CompletionToken {
  std::uint64_t cookie;
  Generation generation;
};

bool generations_equal(const Generation &left,
                       const Generation &right) noexcept {
  return std::memcmp(left.data(), right.data(), kGenerationBytes) == 0;
}

std::uint64_t generation_word(const Generation &generation,
                              std::size_t offset) noexcept {
  std::uint64_t word = 0;
  std::memcpy(&word, generation.data() + offset, sizeof(word));
  return word;
}

Generation generation_from_words(std::uint64_t first,
                                 std::uint64_t second) noexcept {
  Generation generation{};
  std::memcpy(generation.data(), &first, sizeof(first));
  std::memcpy(generation.data() + sizeof(first), &second, sizeof(second));
  return generation;
}

Generation generation_from_python(const py::bytes &value) {
  const std::string bytes = value;
  if (bytes.size() != kGenerationBytes) {
    throw py::value_error("generation must contain exactly 16 bytes");
  }
  Generation generation{};
  std::memcpy(generation.data(), bytes.data(), kGenerationBytes);
  return generation;
}

py::bytes generation_to_python(const Generation &generation) {
  return py::bytes(reinterpret_cast<const char *>(generation.data()),
                   generation.size());
}

template <typename Value> class BoundedMpscQueue {
public:
  explicit BoundedMpscQueue(std::size_t capacity)
      : capacity_(capacity), slots_(std::make_unique<Slot[]>(capacity)) {
    if (capacity < 2) {
      throw std::invalid_argument("queue capacity must be at least two");
    }
    for (std::size_t index = 0; index < capacity_; ++index) {
      slots_[index].sequence.store(index, std::memory_order_relaxed);
    }
  }

  BoundedMpscQueue(const BoundedMpscQueue &) = delete;
  BoundedMpscQueue &operator=(const BoundedMpscQueue &) = delete;

  bool try_enqueue(const Value &value) noexcept {
    std::size_t position = enqueue_position_.load(std::memory_order_relaxed);
    Slot *slot = nullptr;
    for (;;) {
      slot = &slots_[position % capacity_];
      const std::size_t sequence =
          slot->sequence.load(std::memory_order_acquire);
      const auto difference = static_cast<std::intptr_t>(sequence) -
                              static_cast<std::intptr_t>(position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(
                position, position + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          break;
        }
        continue;
      }
      if (difference < 0) {
        return false;
      }
      position = enqueue_position_.load(std::memory_order_relaxed);
    }

    slot->value = value;
    queued_count_.fetch_add(1, std::memory_order_relaxed);
    slot->sequence.store(position + 1, std::memory_order_release);
    return true;
  }

  bool try_dequeue(Value &value) noexcept {
    Slot &slot = slots_[dequeue_position_ % capacity_];
    const std::size_t sequence = slot.sequence.load(std::memory_order_acquire);
    const auto difference = static_cast<std::intptr_t>(sequence) -
                            static_cast<std::intptr_t>(dequeue_position_ + 1);
    if (difference != 0) {
      return false;
    }

    value = slot.value;
    slot.sequence.store(dequeue_position_ + capacity_,
                        std::memory_order_release);
    ++dequeue_position_;
    queued_count_.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }

  std::size_t capacity() const noexcept { return capacity_; }

  std::size_t queued_count() const noexcept {
    return queued_count_.load(std::memory_order_acquire);
  }

private:
  struct Slot {
    std::atomic<std::size_t> sequence{0};
    Value value{};
  };

  const std::size_t capacity_;
  std::unique_ptr<Slot[]> slots_;
  alignas(64) std::atomic<std::size_t> enqueue_position_{0};
  alignas(64) std::size_t dequeue_position_{0};
  std::atomic<std::size_t> queued_count_{0};
};

struct FatalSnapshot {
  FatalCode code{FatalCode::kNone};
  int system_error{0};
  CompletionToken token{};
  bool has_token{false};
};

enum class IdentityPhase : std::uint8_t { kArmed, kSubmitted };

struct IdentityEntry {
  Generation generation;
  IdentityPhase phase;
};

class CompletionState {
public:
  explicit CompletionState(std::size_t capacity)
      : queue_(capacity), event_fd_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
    if (event_fd_.load(std::memory_order_relaxed) < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "eventfd creation failed");
    }
  }

  ~CompletionState() {
    const int fd = event_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
      ::close(fd);
    }
  }

  CompletionState(const CompletionState &) = delete;
  CompletionState &operator=(const CompletionState &) = delete;

  void arm(const CompletionToken &token) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    throw_if_fatal_locked();
    require_open_admission_locked(token);

    const auto existing = identities_.find(token.cookie);
    if (existing != identities_.end()) {
      const FatalCode code =
          generations_equal(existing->second.generation, token.generation)
              ? FatalCode::kDuplicateIdentity
              : FatalCode::kExactGenerationMismatch;
      set_fatal(code, 0, token);
      throw_fatal(code);
    }
    identities_.emplace(token.cookie,
                        IdentityEntry{token.generation, IdentityPhase::kArmed});
  }

  void begin_submission(const CompletionToken &token) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    throw_if_fatal_locked();
    require_open_admission_locked(token);

    const auto existing = identities_.find(token.cookie);
    if (existing == identities_.end()) {
      set_fatal(FatalCode::kSubmissionWithoutArm, 0, token);
      throw_fatal(FatalCode::kSubmissionWithoutArm);
    }
    if (!generations_equal(existing->second.generation, token.generation)) {
      set_fatal(FatalCode::kExactGenerationMismatch, 0, token);
      throw_fatal(FatalCode::kExactGenerationMismatch);
    }
    if (existing->second.phase != IdentityPhase::kArmed) {
      set_fatal(FatalCode::kInvalidIdentityState, 0, token);
      throw_fatal(FatalCode::kInvalidIdentityState);
    }

    existing->second.phase = IdentityPhase::kSubmitted;
    active_callbacks_.fetch_add(1, std::memory_order_release);
    active_registrations_.fetch_add(1, std::memory_order_release);
    producers_joined_.store(false, std::memory_order_release);
  }

  void callback_registration_succeeded() noexcept {
    total_submissions_.fetch_add(1, std::memory_order_relaxed);
    active_registrations_.fetch_sub(1, std::memory_order_release);
  }

  void callback_registration_failed(const CompletionToken &token,
                                    cudaError_t error) noexcept {
    set_fatal(FatalCode::kCudaCallbackRegistrationFailure,
              static_cast<int>(error), token);
    active_callbacks_.fetch_sub(1, std::memory_order_release);
    active_registrations_.fetch_sub(1, std::memory_order_release);
  }

  void publish_from_callback(const CompletionToken &token) noexcept {
    if (queue_.try_enqueue(token)) {
      total_enqueued_.fetch_add(1, std::memory_order_relaxed);
      signal_eventfd();
      active_callbacks_.fetch_sub(1, std::memory_order_release);
      return;
    }

    overflow_count_.fetch_add(1, std::memory_order_relaxed);
    set_fatal(FatalCode::kQueueOverflow, 0, token);
    active_callbacks_.fetch_sub(1, std::memory_order_release);
  }

  int fileno() const noexcept {
    return event_fd_.load(std::memory_order_acquire);
  }

  py::dict drain() {
    bool expected = false;
    if (!drain_active_.compare_exchange_strong(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
      set_fatal(FatalCode::kConcurrentDrain, 0, CompletionToken{}, false);
      throw_fatal(FatalCode::kConcurrentDrain);
    }
    DrainGuard drain_guard(drain_active_);
    if (closed_.load(std::memory_order_acquire)) {
      throw std::runtime_error("CUDA completion bridge is closed");
    }

    const std::uint64_t wake_count = consume_eventfd();
    std::vector<CompletionToken> tokens;
    tokens.reserve(queue_.queued_count());

    CompletionToken token{};
    while (queue_.try_dequeue(token)) {
      if (!retire_identity(token)) {
        continue;
      }
      tokens.push_back(token);
      total_drained_.fetch_add(1, std::memory_order_relaxed);
    }

    py::list token_values;
    for (const CompletionToken &completed : tokens) {
      token_values.append(py::make_tuple(
          completed.cookie, generation_to_python(completed.generation)));
    }

    py::dict result;
    result["wake_count"] = wake_count;
    result["tokens"] = std::move(token_values);
    result["inventory"] = inventory();
    return result;
  }

  void stop_submissions() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    admission_open_ = false;
  }

  bool join_producers() {
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      admission_open_ = false;
    }
    if (active_callbacks_.load(std::memory_order_acquire) != 0 ||
        active_registrations_.load(std::memory_order_acquire) != 0) {
      return false;
    }
    producers_joined_.store(true, std::memory_order_release);
    return true;
  }

  void close() {
    FatalCode close_failure = FatalCode::kNone;
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      admission_open_ = false;
      if (closed_.load(std::memory_order_acquire)) {
        return;
      }
      if (active_callbacks_.load(std::memory_order_acquire) != 0 ||
          active_registrations_.load(std::memory_order_acquire) != 0) {
        close_failure = FatalCode::kCloseWithActiveCallbacks;
      } else if (queue_.queued_count() != 0 || !identities_.empty()) {
        close_failure = FatalCode::kCloseWithRetainedInventory;
      } else if (total_submissions_.load(std::memory_order_acquire) != 0 &&
                 !producers_joined_.load(std::memory_order_acquire)) {
        close_failure = FatalCode::kCloseBeforeProducerJoin;
      }
    }

    if (close_failure != FatalCode::kNone) {
      set_fatal(close_failure, 0, CompletionToken{}, false);
      throw_fatal(close_failure);
    }
    throw_if_fatal();

    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    bool expected = false;
    if (!drain_active_.compare_exchange_strong(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
      set_fatal(FatalCode::kConcurrentDrain, 0, CompletionToken{}, false);
      throw_fatal(FatalCode::kConcurrentDrain);
    }
    DrainGuard close_guard(drain_active_);
    const int fd = event_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
      ::close(fd);
    }
    closed_.store(true, std::memory_order_release);
  }

  py::dict inventory() const {
    std::size_t armed_count = 0;
    std::size_t submitted_identity_count = 0;
    std::size_t live_count = 0;
    bool admission_open = false;
    bool closed = false;
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      live_count = identities_.size();
      for (const auto &[cookie, entry] : identities_) {
        static_cast<void>(cookie);
        if (entry.phase == IdentityPhase::kArmed) {
          ++armed_count;
        } else {
          ++submitted_identity_count;
        }
      }
      admission_open = admission_open_;
      closed = closed_.load(std::memory_order_acquire);
    }

    const FatalSnapshot fatal = fatal_snapshot();
    py::dict result;
    result["capacity"] = queue_.capacity();
    result["armed_count"] = armed_count;
    result["submitted_identity_count"] = submitted_identity_count;
    result["active_callback_count"] =
        active_callbacks_.load(std::memory_order_acquire);
    result["active_registration_count"] =
        active_registrations_.load(std::memory_order_acquire);
    result["queued_count"] = queue_.queued_count();
    result["live_count"] = live_count;
    result["total_submissions"] =
        total_submissions_.load(std::memory_order_acquire);
    result["total_enqueued"] = total_enqueued_.load(std::memory_order_acquire);
    result["total_drained"] = total_drained_.load(std::memory_order_acquire);
    result["overflow_count"] = overflow_count_.load(std::memory_order_acquire);
    result["eventfd_failure_count"] =
        eventfd_failure_count_.load(std::memory_order_acquire);
    result["successful_wake_count"] =
        successful_wake_count_.load(std::memory_order_acquire);
    result["consumed_wake_count"] =
        consumed_wake_count_.load(std::memory_order_acquire);
    result["rejected_token_count"] =
        rejected_token_count_.load(std::memory_order_acquire);
    result["producers_joined"] =
        producers_joined_.load(std::memory_order_acquire);
    result["admission_open"] = admission_open;
    result["closed"] = closed;
    result["eventfd_open"] = fileno() >= 0;
    result["fatal_code"] = fatal_code_name(fatal.code);
    result["fatal_system_error"] = fatal.system_error;
    if (fatal.has_token) {
      result["fatal_cookie"] = fatal.token.cookie;
      result["fatal_generation"] = generation_to_python(fatal.token.generation);
    } else {
      result["fatal_cookie"] = py::none();
      result["fatal_generation"] = py::none();
    }
    return result;
  }

#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
  void complete_synchronously_for_test(const CompletionToken &token) {
    begin_submission(token);
    callback_registration_succeeded();
    publish_from_callback(token);
  }

  void break_eventfd_for_test() {
    const int fd = event_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
      ::close(fd);
    }
  }
#endif

private:
  class DrainGuard {
  public:
    explicit DrainGuard(std::atomic<bool> &flag) : flag_(flag) {}
    ~DrainGuard() { flag_.store(false, std::memory_order_release); }

    DrainGuard(const DrainGuard &) = delete;
    DrainGuard &operator=(const DrainGuard &) = delete;

  private:
    std::atomic<bool> &flag_;
  };

  void require_open_admission_locked(const CompletionToken &token) {
    if (admission_open_ && !closed_.load(std::memory_order_acquire)) {
      return;
    }
    set_fatal(FatalCode::kSubmissionAfterStop, 0, token);
    throw_fatal(FatalCode::kSubmissionAfterStop);
  }

  bool retire_identity(const CompletionToken &token) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    const auto existing = identities_.find(token.cookie);
    if (existing == identities_.end()) {
      rejected_token_count_.fetch_add(1, std::memory_order_relaxed);
      set_fatal(FatalCode::kInvalidIdentityState, 0, token);
      return false;
    }
    if (!generations_equal(existing->second.generation, token.generation)) {
      rejected_token_count_.fetch_add(1, std::memory_order_relaxed);
      set_fatal(FatalCode::kExactGenerationMismatch, 0, token);
      return false;
    }
    if (existing->second.phase != IdentityPhase::kSubmitted) {
      rejected_token_count_.fetch_add(1, std::memory_order_relaxed);
      set_fatal(FatalCode::kInvalidIdentityState, 0, token);
      return false;
    }
    identities_.erase(existing);
    return true;
  }

  std::uint64_t consume_eventfd() {
    const int fd = fileno();
    if (fd < 0) {
      if (!closed_.load(std::memory_order_acquire)) {
        eventfd_failure_count_.fetch_add(1, std::memory_order_relaxed);
        set_fatal(FatalCode::kEventfdReadFailure, EBADF, CompletionToken{},
                  false);
      }
      return 0;
    }

    std::uint64_t wake_count = 0;
    for (;;) {
      std::uint64_t observed = 0;
      const ssize_t result = ::read(fd, &observed, sizeof(observed));
      if (result == static_cast<ssize_t>(sizeof(observed))) {
        wake_count += observed;
        continue;
      }
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
      }
      const int error = result < 0 ? errno : EIO;
      eventfd_failure_count_.fetch_add(1, std::memory_order_relaxed);
      set_fatal(FatalCode::kEventfdReadFailure, error, CompletionToken{},
                false);
      break;
    }
    consumed_wake_count_.fetch_add(wake_count, std::memory_order_relaxed);
    return wake_count;
  }

  bool signal_eventfd_raw() noexcept {
    const int fd = fileno();
    if (fd < 0) {
      errno = EBADF;
      return false;
    }
    constexpr std::uint64_t increment = 1;
    for (;;) {
      const ssize_t result = ::write(fd, &increment, sizeof(increment));
      if (result == static_cast<ssize_t>(sizeof(increment))) {
        successful_wake_count_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      if (result < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }
  }

  void signal_eventfd() noexcept {
    if (signal_eventfd_raw()) {
      return;
    }
    const int error = errno;
    eventfd_failure_count_.fetch_add(1, std::memory_order_relaxed);
    set_fatal(FatalCode::kEventfdWriteFailure, error, CompletionToken{}, false,
              false);
  }

  void set_fatal(FatalCode code, int system_error, const CompletionToken &token,
                 bool has_token = true, bool signal = true) const noexcept {
    std::uint32_t expected = 0;
    if (fatal_publication_state_.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      fatal_code_.store(static_cast<std::uint32_t>(code),
                        std::memory_order_relaxed);
      fatal_system_error_.store(system_error, std::memory_order_relaxed);
      fatal_cookie_.store(token.cookie, std::memory_order_relaxed);
      fatal_generation_first_.store(generation_word(token.generation, 0),
                                    std::memory_order_relaxed);
      fatal_generation_second_.store(generation_word(token.generation, 8),
                                     std::memory_order_relaxed);
      fatal_has_token_.store(has_token, std::memory_order_relaxed);
      fatal_publication_state_.store(2, std::memory_order_release);
    }
    if (signal) {
      const_cast<CompletionState *>(this)->signal_eventfd();
    }
  }

  FatalSnapshot fatal_snapshot() const noexcept {
    std::uint32_t publication =
        fatal_publication_state_.load(std::memory_order_acquire);
    while (publication == 1) {
      publication = fatal_publication_state_.load(std::memory_order_acquire);
    }
    if (publication == 0) {
      return FatalSnapshot{};
    }
    FatalSnapshot snapshot;
    snapshot.code =
        static_cast<FatalCode>(fatal_code_.load(std::memory_order_relaxed));
    snapshot.system_error = fatal_system_error_.load(std::memory_order_relaxed);
    snapshot.token.cookie = fatal_cookie_.load(std::memory_order_relaxed);
    snapshot.token.generation = generation_from_words(
        fatal_generation_first_.load(std::memory_order_relaxed),
        fatal_generation_second_.load(std::memory_order_relaxed));
    snapshot.has_token = fatal_has_token_.load(std::memory_order_relaxed);
    return snapshot;
  }

  void throw_if_fatal() const {
    const FatalSnapshot fatal = fatal_snapshot();
    if (fatal.code != FatalCode::kNone) {
      throw_fatal(fatal.code);
    }
  }

  void throw_if_fatal_locked() const { throw_if_fatal(); }

  [[noreturn]] static void throw_fatal(FatalCode code) {
    throw std::runtime_error(std::string("CUDA completion bridge fatal: ") +
                             fatal_code_name(code));
  }

  BoundedMpscQueue<CompletionToken> queue_;
  std::atomic<int> event_fd_;
  mutable std::mutex lifecycle_mutex_;
  std::unordered_map<std::uint64_t, IdentityEntry> identities_;
  bool admission_open_{true};
  std::atomic<bool> closed_{false};
  std::atomic<std::size_t> active_callbacks_{0};
  std::atomic<std::size_t> active_registrations_{0};
  std::atomic<bool> producers_joined_{false};
  std::atomic<bool> drain_active_{false};
  std::atomic<std::uint64_t> total_submissions_{0};
  std::atomic<std::uint64_t> total_enqueued_{0};
  std::atomic<std::uint64_t> total_drained_{0};
  std::atomic<std::uint64_t> overflow_count_{0};
  std::atomic<std::uint64_t> eventfd_failure_count_{0};
  std::atomic<std::uint64_t> successful_wake_count_{0};
  std::atomic<std::uint64_t> consumed_wake_count_{0};
  std::atomic<std::uint64_t> rejected_token_count_{0};
  mutable std::atomic<std::uint32_t> fatal_publication_state_{0};
  mutable std::atomic<std::uint32_t> fatal_code_{0};
  mutable std::atomic<int> fatal_system_error_{0};
  mutable std::atomic<std::uint64_t> fatal_cookie_{0};
  mutable std::atomic<std::uint64_t> fatal_generation_first_{0};
  mutable std::atomic<std::uint64_t> fatal_generation_second_{0};
  mutable std::atomic<bool> fatal_has_token_{false};
};

struct CallbackPayload {
  std::shared_ptr<CompletionState> state;
  CompletionToken token;
};

void CUDART_CB completion_callback(void *opaque) noexcept {
  std::unique_ptr<CallbackPayload> payload(
      static_cast<CallbackPayload *>(opaque));
  payload->state->publish_from_callback(payload->token);
}

class NativeCudaCompletionBridge {
public:
  explicit NativeCudaCompletionBridge(std::size_t capacity)
      : state_(std::make_shared<CompletionState>(capacity)) {}

  void arm(std::uint64_t cookie, const py::bytes &generation) {
    state_->arm(CompletionToken{cookie, generation_from_python(generation)});
  }

  void submit(std::uintptr_t stream_handle, std::uint64_t cookie,
              const py::bytes &generation) {
    const CompletionToken token{cookie, generation_from_python(generation)};
    auto payload =
        std::make_unique<CallbackPayload>(CallbackPayload{state_, token});
    state_->begin_submission(token);

    const cudaError_t result =
        cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream_handle),
                           completion_callback, payload.get());
    if (result != cudaSuccess) {
      state_->callback_registration_failed(token, result);
      throw std::runtime_error(std::string("cudaLaunchHostFunc failed: ") +
                               cudaGetErrorName(result) + ": " +
                               cudaGetErrorString(result));
    }
    state_->callback_registration_succeeded();
    static_cast<void>(payload.release());
  }

  int fileno() const noexcept { return state_->fileno(); }

  py::dict drain() { return state_->drain(); }

  void stop_submissions() { state_->stop_submissions(); }

  bool join_producers() { return state_->join_producers(); }

  void close() { state_->close(); }

  py::dict inventory() const { return state_->inventory(); }

#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
  void complete_synchronously_for_test(std::uint64_t cookie,
                                       const py::bytes &generation) {
    state_->complete_synchronously_for_test(
        CompletionToken{cookie, generation_from_python(generation)});
  }

  void break_eventfd_for_test() { state_->break_eventfd_for_test(); }
#endif

private:
  std::shared_ptr<CompletionState> state_;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<NativeCudaCompletionBridge>(module, "CudaCompletionBridge")
      .def(py::init<std::size_t>(), py::arg("capacity"))
      .def("arm", &NativeCudaCompletionBridge::arm, py::arg("cookie"),
           py::arg("generation"))
      .def("submit", &NativeCudaCompletionBridge::submit,
           py::arg("stream_handle"), py::arg("cookie"), py::arg("generation"))
      .def("fileno", &NativeCudaCompletionBridge::fileno)
      .def("drain", &NativeCudaCompletionBridge::drain)
      .def("stop_submissions", &NativeCudaCompletionBridge::stop_submissions)
      .def("join_producers", &NativeCudaCompletionBridge::join_producers)
      .def("close", &NativeCudaCompletionBridge::close)
      .def("inventory", &NativeCudaCompletionBridge::inventory)
#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
      .def("_complete_synchronously_for_test",
           &NativeCudaCompletionBridge::complete_synchronously_for_test,
           py::arg("cookie"), py::arg("generation"))
      .def("_break_eventfd_for_test",
           &NativeCudaCompletionBridge::break_eventfd_for_test)
#endif
      ;
}
