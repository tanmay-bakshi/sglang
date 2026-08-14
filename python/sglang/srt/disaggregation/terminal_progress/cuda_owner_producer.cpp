#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "native_producer_api.h"

namespace py = pybind11;

namespace {

constexpr std::size_t kDigestBytes = 32;
constexpr std::uint16_t kSourceProducerCompletedEvent = 11;
constexpr std::uint16_t kDecodeScatterTerminalEvent = 44;

std::uint16_t require_event_kind(std::uint16_t event_kind) {
  if (event_kind != kSourceProducerCompletedEvent &&
      event_kind != kDecodeScatterTerminalEvent) {
    throw std::invalid_argument("unsupported CUDA terminal event kind");
  }
  return event_kind;
}

using Digest = std::array<std::uint8_t, kDigestBytes>;

enum class BindingPhase : std::uint8_t {
  kArmed = 1,
  kSubmitted = 2,
};

enum class FatalCode : std::uint32_t {
  kNone = 0,
  kDuplicateBinding = 1,
  kUnknownBinding = 2,
  kInvalidBindingState = 3,
  kCudaCallbackRegistrationFailure = 4,
  kOwnerSubmissionFailure = 5,
  kClockFailure = 6,
  kSubmissionAfterStop = 7,
  kRetirementFailure = 8,
  kJoinFailure = 9,
  kCloseWithActiveCallbacks = 10,
  kCloseWithRetainedBindings = 11,
  kCloseBeforeProducerJoin = 12,
};

const char *fatal_code_name(FatalCode code) noexcept {
  switch (code) {
  case FatalCode::kNone:
    return "none";
  case FatalCode::kDuplicateBinding:
    return "duplicate_binding";
  case FatalCode::kUnknownBinding:
    return "unknown_binding";
  case FatalCode::kInvalidBindingState:
    return "invalid_binding_state";
  case FatalCode::kCudaCallbackRegistrationFailure:
    return "cuda_callback_registration_failure";
  case FatalCode::kOwnerSubmissionFailure:
    return "owner_submission_failure";
  case FatalCode::kClockFailure:
    return "clock_failure";
  case FatalCode::kSubmissionAfterStop:
    return "submission_after_stop";
  case FatalCode::kRetirementFailure:
    return "retirement_failure";
  case FatalCode::kJoinFailure:
    return "join_failure";
  case FatalCode::kCloseWithActiveCallbacks:
    return "close_with_active_callbacks";
  case FatalCode::kCloseWithRetainedBindings:
    return "close_with_retained_bindings";
  case FatalCode::kCloseBeforeProducerJoin:
    return "close_before_producer_join";
  }
  return "unknown";
}

struct DigestHash {
  std::size_t operator()(const Digest &value) const noexcept {
    std::size_t hash = 1469598103934665603ULL;
    for (const std::uint8_t byte : value) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

Digest digest_from_python(const py::bytes &value) {
  const std::string bytes = value;
  if (bytes.size() != kDigestBytes) {
    throw py::value_error("binding digest must contain exactly 32 bytes");
  }
  Digest digest{};
  std::memcpy(digest.data(), bytes.data(), digest.size());
  return digest;
}

py::bytes digest_to_python(const Digest &value) {
  return py::bytes(reinterpret_cast<const char *>(value.data()), value.size());
}

std::uint64_t monotonic_raw_ns() noexcept {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

class ProducerState {
public:
  ProducerState(const sglang_terminal_owner_producer_api_v1 *api, void *context,
                std::uint16_t event_kind)
      : api_(api), context_(context), event_kind_(event_kind) {}

  void arm(const Digest &binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_healthy_locked();
    if (!admission_open_) {
      set_fatal_locked(FatalCode::kSubmissionAfterStop, ESHUTDOWN, binding);
      throw_fatal_locked();
    }
    const auto [iterator, inserted] =
        bindings_.emplace(binding, BindingPhase::kArmed);
    static_cast<void>(iterator);
    if (!inserted) {
      set_fatal_locked(FatalCode::kDuplicateBinding, EEXIST, binding);
      throw_fatal_locked();
    }
  }

  void begin_submission(const Digest &binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_healthy_locked();
    if (!admission_open_) {
      set_fatal_locked(FatalCode::kSubmissionAfterStop, ESHUTDOWN, binding);
      throw_fatal_locked();
    }
    const auto iterator = bindings_.find(binding);
    if (iterator == bindings_.end()) {
      set_fatal_locked(FatalCode::kUnknownBinding, ENOENT, binding);
      throw_fatal_locked();
    }
    if (iterator->second != BindingPhase::kArmed) {
      set_fatal_locked(FatalCode::kInvalidBindingState, EALREADY, binding);
      throw_fatal_locked();
    }
    iterator->second = BindingPhase::kSubmitted;
    active_callbacks_.fetch_add(1, std::memory_order_release);
    active_registrations_.fetch_add(1, std::memory_order_release);
  }

  void callback_registration_succeeded() noexcept {
    total_submissions_.fetch_add(1, std::memory_order_relaxed);
    active_registrations_.fetch_sub(1, std::memory_order_release);
    condition_.notify_all();
  }

  void callback_registration_failed(const Digest &binding,
                                    cudaError_t status) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      set_fatal_locked(FatalCode::kCudaCallbackRegistrationFailure,
                       static_cast<int>(status), binding);
    }
    active_callbacks_.fetch_sub(1, std::memory_order_release);
    active_registrations_.fetch_sub(1, std::memory_order_release);
    condition_.notify_all();
  }

  void publish_from_callback(
      sglang_terminal_owner_producer_event_v1 event) noexcept {
    const Digest binding = event_binding(event);
    const std::uint64_t timestamp_ns = monotonic_raw_ns();
    if (timestamp_ns == 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      set_fatal_locked(FatalCode::kClockFailure, errno, binding);
      finish_callback();
      return;
    }
    event.enqueued_ns = timestamp_ns;
    const int status = api_->submit(context_, &event);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (status != 0) {
        owner_submit_failure_count_.fetch_add(1, std::memory_order_relaxed);
        set_fatal_locked(FatalCode::kOwnerSubmissionFailure, status, binding);
      } else {
        const auto iterator = bindings_.find(binding);
        if (iterator == bindings_.end() ||
            iterator->second != BindingPhase::kSubmitted) {
          set_fatal_locked(FatalCode::kInvalidBindingState, EPROTO, binding);
        } else {
          bindings_.erase(iterator);
          total_delivered_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    finish_callback();
  }

  void stop_admission() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    admission_open_ = false;
  }

  bool join(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("producer join timeout must be positive");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      admission_open_ = false;
      require_healthy_locked();
      if (active_callbacks_.load(std::memory_order_acquire) != 0 ||
          active_registrations_.load(std::memory_order_acquire) != 0) {
        return false;
      }
      if (!bindings_.empty()) {
        set_fatal_locked(FatalCode::kCloseWithRetainedBindings, EBUSY,
                         bindings_.begin()->first);
        throw_fatal_locked();
      }
      if (!retirement_requested_) {
        const int retire_status = api_->retire(context_);
        if (retire_status != 0) {
          set_fatal_locked(FatalCode::kRetirementFailure, retire_status,
                           Digest{});
          throw_fatal_locked();
        }
        retirement_requested_ = true;
      }
    }

    const std::uint64_t timeout_ns = static_cast<std::uint64_t>(
        timeout_seconds * static_cast<double>(1'000'000'000ULL));
    int join_status = 0;
    {
      py::gil_scoped_release release;
      join_status = api_->join(context_, timeout_ns);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (join_status == ETIMEDOUT) {
      return false;
    }
    if (join_status != 0) {
      set_fatal_locked(FatalCode::kJoinFailure, join_status, Digest{});
      throw_fatal_locked();
    }
    joined_ = true;
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    admission_open_ = false;
    if (closed_) {
      return;
    }
    if (active_callbacks_.load(std::memory_order_acquire) != 0 ||
        active_registrations_.load(std::memory_order_acquire) != 0) {
      set_fatal_locked(FatalCode::kCloseWithActiveCallbacks, EBUSY, Digest{});
      throw_fatal_locked();
    }
    if (!bindings_.empty()) {
      set_fatal_locked(FatalCode::kCloseWithRetainedBindings, EBUSY,
                       bindings_.begin()->first);
      throw_fatal_locked();
    }
    if (!joined_) {
      set_fatal_locked(FatalCode::kCloseBeforeProducerJoin, EBUSY, Digest{});
      throw_fatal_locked();
    }
    require_healthy_locked();
    closed_ = true;
  }

  py::dict inventory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t armed_count = 0;
    std::size_t submitted_count = 0;
    for (const auto &[binding, phase] : bindings_) {
      static_cast<void>(binding);
      if (phase == BindingPhase::kArmed) {
        ++armed_count;
      } else {
        ++submitted_count;
      }
    }
    py::dict result;
    result["armed_count"] = armed_count;
    result["submitted_count"] = submitted_count;
    result["active_callback_count"] =
        active_callbacks_.load(std::memory_order_acquire);
    result["active_registration_count"] =
        active_registrations_.load(std::memory_order_acquire);
    result["total_submissions"] =
        total_submissions_.load(std::memory_order_acquire);
    result["total_delivered"] =
        total_delivered_.load(std::memory_order_acquire);
    result["owner_submit_failure_count"] =
        owner_submit_failure_count_.load(std::memory_order_acquire);
    result["admission_open"] = admission_open_;
    result["retirement_requested"] = retirement_requested_;
    result["joined"] = joined_;
    result["closed"] = closed_;
    result["fatal_code"] = fatal_code_name(fatal_code_);
    result["fatal_status"] = fatal_status_;
    if (fatal_has_binding_) {
      result["fatal_binding"] = digest_to_python(fatal_binding_);
    } else {
      result["fatal_binding"] = py::none();
    }
    return result;
  }

#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
  void complete_synchronously_for_test(const Digest &binding) {
    begin_submission(binding);
    callback_registration_succeeded();
    publish_from_callback(make_event(binding));
  }

  void begin_held_callback_for_test(const Digest &binding) {
    begin_submission(binding);
    callback_registration_succeeded();
  }

  void complete_held_callback_for_test(const Digest &binding) {
    publish_from_callback(make_event(binding));
  }

  void complete_concurrently_for_test(const std::vector<Digest> &bindings) {
    for (const Digest &binding : bindings) {
      begin_submission(binding);
      callback_registration_succeeded();
    }
    std::vector<std::thread> producers;
    producers.reserve(bindings.size());
    for (const Digest &binding : bindings) {
      producers.emplace_back(
          [this, binding]() { publish_from_callback(make_event(binding)); });
    }
    for (std::thread &producer : producers) {
      producer.join();
    }
  }
#endif

private:
  static Digest
  event_binding(const sglang_terminal_owner_producer_event_v1 &event) noexcept {
    Digest binding{};
    std::memcpy(binding.data(), event.binding_digest, binding.size());
    return binding;
  }

  sglang_terminal_owner_producer_event_v1
  make_event(const Digest &binding) const noexcept {
    sglang_terminal_owner_producer_event_v1 event{};
    event.abi_version = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION;
    event.struct_size = sizeof(event);
    std::memcpy(event.binding_digest, binding.data(), binding.size());
    event.event_kind = event_kind_;
    return event;
  }

  void finish_callback() noexcept {
    active_callbacks_.fetch_sub(1, std::memory_order_release);
    condition_.notify_all();
  }

  void require_healthy_locked() const {
    if (fatal_code_ != FatalCode::kNone) {
      throw_fatal_locked();
    }
    if (closed_) {
      throw std::runtime_error("CUDA terminal producer is closed");
    }
  }

  void set_fatal_locked(FatalCode code, int status,
                        const Digest &binding) noexcept {
    if (fatal_code_ != FatalCode::kNone) {
      return;
    }
    fatal_code_ = code;
    fatal_status_ = status;
    fatal_binding_ = binding;
    fatal_has_binding_ = binding != Digest{};
    admission_open_ = false;
  }

  [[noreturn]] void throw_fatal_locked() const {
    throw std::runtime_error(std::string("CUDA terminal producer fatal: ") +
                             fatal_code_name(fatal_code_));
  }

  const sglang_terminal_owner_producer_api_v1 *api_;
  void *context_;
  std::uint16_t event_kind_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::unordered_map<Digest, BindingPhase, DigestHash> bindings_;
  bool admission_open_{true};
  bool retirement_requested_{false};
  bool joined_{false};
  bool closed_{false};
  FatalCode fatal_code_{FatalCode::kNone};
  int fatal_status_{0};
  Digest fatal_binding_{};
  bool fatal_has_binding_{false};
  std::atomic<std::size_t> active_callbacks_{0};
  std::atomic<std::size_t> active_registrations_{0};
  std::atomic<std::uint64_t> total_submissions_{0};
  std::atomic<std::uint64_t> total_delivered_{0};
  std::atomic<std::uint64_t> owner_submit_failure_count_{0};
};

struct CallbackPayload {
  std::shared_ptr<ProducerState> state;
  sglang_terminal_owner_producer_event_v1 event;
};

void CUDART_CB completion_callback(void *opaque) noexcept {
  std::unique_ptr<CallbackPayload> payload(
      static_cast<CallbackPayload *>(opaque));
  payload->state->publish_from_callback(payload->event);
}

std::mutex abandoned_capsules_mutex;
std::vector<std::pair<PyObject *, PyObject *>> abandoned_capsules;

class NativeCudaTerminalProducer {
public:
  NativeCudaTerminalProducer(const py::capsule &api_capsule,
                             const py::capsule &context_capsule,
                             std::uint16_t event_kind)
      : api_capsule_(api_capsule.ptr()),
        context_capsule_(context_capsule.ptr()),
        event_kind_(require_event_kind(event_kind)) {
    const auto *api =
        static_cast<const sglang_terminal_owner_producer_api_v1 *>(
            PyCapsule_GetPointer(
                api_capsule.ptr(),
                SGLANG_TERMINAL_OWNER_PRODUCER_API_CAPSULE_NAME));
    if (api == nullptr) {
      throw py::error_already_set();
    }
    void *context = PyCapsule_GetPointer(
        context_capsule.ptr(),
        SGLANG_TERMINAL_OWNER_PRODUCER_CONTEXT_CAPSULE_NAME);
    if (context == nullptr) {
      throw py::error_already_set();
    }
    if (api->abi_version != SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION ||
        api->struct_size != sizeof(sglang_terminal_owner_producer_api_v1) ||
        api->event_struct_size !=
            sizeof(sglang_terminal_owner_producer_event_v1) ||
        (api->flags & SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS) !=
            SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS ||
        api->submit == nullptr || api->retire == nullptr ||
        api->join == nullptr) {
      throw std::invalid_argument("terminal owner producer ABI mismatch");
    }
    state_ = std::make_shared<ProducerState>(api, context, event_kind_);
    Py_INCREF(api_capsule_);
    Py_INCREF(context_capsule_);
  }

  ~NativeCudaTerminalProducer() {
    if (api_capsule_ == nullptr || context_capsule_ == nullptr) {
      return;
    }
    const py::dict inventory = state_->inventory();
    if (py::cast<bool>(inventory["closed"])) {
      Py_DECREF(api_capsule_);
      Py_DECREF(context_capsule_);
    } else {
      // An abandoned live callback may still dereference the owner context.
      // Quarantine both capsule references for the process lifetime rather
      // than turning a lifecycle failure into a use-after-free.
      std::lock_guard<std::mutex> lock(abandoned_capsules_mutex);
      abandoned_capsules.emplace_back(api_capsule_, context_capsule_);
    }
    api_capsule_ = nullptr;
    context_capsule_ = nullptr;
  }

  void arm(const py::bytes &binding_digest) {
    state_->arm(digest_from_python(binding_digest));
  }

  void submit(std::uintptr_t stream_handle, const py::bytes &binding_digest) {
    const Digest binding = digest_from_python(binding_digest);
    auto payload = std::make_unique<CallbackPayload>(
        CallbackPayload{state_, make_event(binding)});
    state_->begin_submission(binding);
    const cudaError_t status =
        cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream_handle),
                           completion_callback, payload.get());
    if (status != cudaSuccess) {
      state_->callback_registration_failed(binding, status);
      throw std::runtime_error(std::string("cudaLaunchHostFunc failed: ") +
                               cudaGetErrorName(status) + ": " +
                               cudaGetErrorString(status));
    }
    state_->callback_registration_succeeded();
    static_cast<void>(payload.release());
  }

  void stop_admission() noexcept { state_->stop_admission(); }

  bool join(double timeout_seconds) { return state_->join(timeout_seconds); }

  void close() {
    state_->close();
    Py_DECREF(api_capsule_);
    Py_DECREF(context_capsule_);
    api_capsule_ = nullptr;
    context_capsule_ = nullptr;
  }

  py::dict inventory() const { return state_->inventory(); }

#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
  void complete_synchronously_for_test(const py::bytes &binding_digest) {
    state_->complete_synchronously_for_test(digest_from_python(binding_digest));
  }

  void begin_held_callback_for_test(const py::bytes &binding_digest) {
    state_->begin_held_callback_for_test(digest_from_python(binding_digest));
  }

  void complete_held_callback_for_test(const py::bytes &binding_digest) {
    state_->complete_held_callback_for_test(digest_from_python(binding_digest));
  }

  void complete_concurrently_for_test(const py::list &values) {
    std::vector<Digest> bindings;
    bindings.reserve(values.size());
    for (const py::handle value : values) {
      bindings.push_back(digest_from_python(py::cast<py::bytes>(value)));
    }
    state_->complete_concurrently_for_test(bindings);
  }
#endif

private:
  sglang_terminal_owner_producer_event_v1
  make_event(const Digest &binding) const noexcept {
    sglang_terminal_owner_producer_event_v1 event{};
    event.abi_version = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION;
    event.struct_size = sizeof(event);
    std::memcpy(event.binding_digest, binding.data(), binding.size());
    event.event_kind = event_kind_;
    return event;
  }

  std::shared_ptr<ProducerState> state_;
  PyObject *api_capsule_{nullptr};
  PyObject *context_capsule_{nullptr};
  std::uint16_t event_kind_;
};

py::dict compiled_abi() {
  py::dict offsets;
  offsets["abi_version"] =
      offsetof(sglang_terminal_owner_producer_event_v1, abi_version);
  offsets["struct_size"] =
      offsetof(sglang_terminal_owner_producer_event_v1, struct_size);
  offsets["binding_digest"] =
      offsetof(sglang_terminal_owner_producer_event_v1, binding_digest);
  offsets["event_kind"] =
      offsetof(sglang_terminal_owner_producer_event_v1, event_kind);
  offsets["enqueued_ns"] =
      offsetof(sglang_terminal_owner_producer_event_v1, enqueued_ns);
  offsets["receipt_binding_digest"] =
      offsetof(sglang_terminal_owner_producer_event_v1, receipt_binding_digest);
  offsets["receipt_nonce"] =
      offsetof(sglang_terminal_owner_producer_event_v1, receipt_nonce);
  py::dict result;
  result["abi_version"] = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION;
  result["api_struct_size"] = sizeof(sglang_terminal_owner_producer_api_v1);
  result["event_struct_size"] = sizeof(sglang_terminal_owner_producer_event_v1);
  result["required_flags"] = SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS;
  result["event_offsets"] = std::move(offsets);
  return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<NativeCudaTerminalProducer>(module, "CudaTerminalProducer")
      .def(py::init<const py::capsule &, const py::capsule &, std::uint16_t>(),
           py::arg("producer_api"), py::arg("producer_context"),
           py::arg("event_kind"))
      .def("arm", &NativeCudaTerminalProducer::arm, py::arg("binding_digest"))
      .def("submit", &NativeCudaTerminalProducer::submit,
           py::arg("stream_handle"), py::arg("binding_digest"))
      .def("stop_admission", &NativeCudaTerminalProducer::stop_admission)
      .def("join", &NativeCudaTerminalProducer::join,
           py::arg("timeout_seconds"))
      .def("close", &NativeCudaTerminalProducer::close)
      .def("inventory", &NativeCudaTerminalProducer::inventory)
#ifdef SGLANG_CUDA_COMPLETION_BRIDGE_TESTING
      .def("_complete_synchronously_for_test",
           &NativeCudaTerminalProducer::complete_synchronously_for_test,
           py::arg("binding_digest"))
      .def("_begin_held_callback_for_test",
           &NativeCudaTerminalProducer::begin_held_callback_for_test,
           py::arg("binding_digest"))
      .def("_complete_held_callback_for_test",
           &NativeCudaTerminalProducer::complete_held_callback_for_test,
           py::arg("binding_digest"))
      .def("_complete_concurrently_for_test",
           &NativeCudaTerminalProducer::complete_concurrently_for_test,
           py::arg("binding_digests"))
#endif
      ;
  module.def("compiled_abi", &compiled_abi);
}
