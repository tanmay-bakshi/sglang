#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <system_error>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

namespace {

enum class FatalCode : std::uint8_t {
  kNone = 0,
  kQueueOverflow = 1,
  kEventfdWriteFailure = 2,
  kEventfdReadFailure = 3,
  kUnknownSequence = 4,
  kInvalidMachinePhase = 5,
  kConcurrentDrain = 6,
  kStartAfterClose = 7,
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
  case FatalCode::kUnknownSequence:
    return "unknown_sequence";
  case FatalCode::kInvalidMachinePhase:
    return "invalid_machine_phase";
  case FatalCode::kConcurrentDrain:
    return "concurrent_drain";
  case FatalCode::kStartAfterClose:
    return "start_after_close";
  }
  return "unknown";
}

std::uint64_t monotonic_raw_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "CLOCK_MONOTONIC_RAW failed");
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

struct EventRecord {
  std::uint64_t producer_sequence{0};
  std::uint32_t machine_index{0};
  std::uint64_t generation_index{0};
  std::uint32_t hop_index{0};
  std::uint64_t enqueued_ns{0};
};

struct HopTrace {
  EventRecord event{};
  std::uint64_t completed_ns{0};
};

enum class MachinePhase : std::uint8_t {
  kInitial,
  kQueued,
  kDelivered,
  kRetired,
};

struct MachineState {
  MachinePhase phase{MachinePhase::kInitial};
  std::uint64_t generation_index{0};
  std::uint32_t hop_index{0};
  EventRecord current{};
};

class NativeGILQualificationBridge {
public:
  NativeGILQualificationBridge(std::size_t machine_count,
                               std::size_t hop_count,
                               std::size_t capacity)
      : machine_count_(machine_count), hop_count_(hop_count),
        capacity_(capacity), machines_(machine_count),
        event_fd_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
    if (machine_count == 0) {
      throw std::invalid_argument("machine_count must be positive");
    }
    if (hop_count == 0) {
      throw std::invalid_argument("hop_count must be positive");
    }
    if (capacity == 0) {
      throw std::invalid_argument("capacity must be positive");
    }
    if (event_fd_ < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "eventfd creation failed");
    }
  }

  ~NativeGILQualificationBridge() { abort_and_close(); }

  NativeGILQualificationBridge(const NativeGILQualificationBridge &) = delete;
  NativeGILQualificationBridge &
  operator=(const NativeGILQualificationBridge &) = delete;

  int fileno() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return event_fd_;
  }

  void start(double minimum_duration_seconds,
             std::uint64_t minimum_transition_count) {
    if (minimum_duration_seconds <= 0.0) {
      throw std::invalid_argument("minimum_duration_seconds must be positive");
    }
    if (minimum_transition_count == 0) {
      throw std::invalid_argument("minimum_transition_count must be positive");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) {
      throw std::runtime_error("qualification bridge cannot restart");
    }
    if (closed_) {
      set_fatal_locked(FatalCode::kStartAfterClose, 0, false);
      throw std::runtime_error("qualification bridge is closed");
    }
    minimum_duration_ns_ = static_cast<std::uint64_t>(
        minimum_duration_seconds * 1'000'000'000.0);
    minimum_transition_count_ = minimum_transition_count;
    started_ns_ = monotonic_raw_ns();
    started_ = true;
    for (std::size_t machine_index = 0; machine_index < machine_count_;
         ++machine_index) {
      enqueue_machine_locked(machine_index);
      if (fatal_code_ != FatalCode::kNone) {
        throw_fatal_locked();
      }
    }
  }

  py::dict drain() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (drain_active_) {
        set_fatal_locked(FatalCode::kConcurrentDrain, 0);
        throw_fatal_locked();
      }
      drain_active_ = true;
    }
    DrainGuard guard(*this);
    const std::uint64_t wake_count = consume_eventfd();

    std::vector<EventRecord> records;
    py::dict inventory_value;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      records.reserve(queue_.size());
      while (!queue_.empty()) {
        const EventRecord record = queue_.front();
        queue_.pop_front();
        MachineState &machine = machines_.at(record.machine_index);
        if (machine.phase != MachinePhase::kQueued ||
            machine.current.producer_sequence != record.producer_sequence) {
          set_fatal_locked(FatalCode::kInvalidMachinePhase, 0);
          break;
        }
        machine.phase = MachinePhase::kDelivered;
        records.push_back(record);
      }
      inventory_value = inventory_locked();
    }

    py::list record_values;
    for (const EventRecord &record : records) {
      record_values.append(event_record_to_python(record));
    }
    py::dict result;
    result["wake_count"] = wake_count;
    result["records"] = std::move(record_values);
    result["inventory"] = std::move(inventory_value);
    return result;
  }

  void acknowledge_dispatch(std::uint64_t producer_sequence) {
    const std::uint64_t completed_ns = monotonic_raw_ns();
    std::lock_guard<std::mutex> lock(mutex_);
    throw_if_fatal_locked();
    MachineState *matched = nullptr;
    for (MachineState &machine : machines_) {
      if (machine.phase == MachinePhase::kDelivered &&
          machine.current.producer_sequence == producer_sequence) {
        matched = &machine;
        break;
      }
    }
    if (matched == nullptr) {
      set_fatal_locked(FatalCode::kUnknownSequence, 0);
      throw_fatal_locked();
    }

    traces_.push_back(HopTrace{matched->current, completed_ns});
    ++transition_count_;
    last_completed_ns_ = completed_ns;
    const bool request_complete = matched->hop_index + 1 == hop_count_;
    if (!draining_ && request_complete &&
        completed_ns - started_ns_ >= minimum_duration_ns_ &&
        transition_count_ >= minimum_transition_count_) {
      draining_ = true;
    }

    if (draining_ && request_complete) {
      matched->phase = MachinePhase::kRetired;
    } else {
      if (request_complete) {
        ++matched->generation_index;
        matched->hop_index = 0;
      } else {
        ++matched->hop_index;
      }
      enqueue_machine_locked(matched->current.machine_index);
      throw_if_fatal_locked();
    }

    bool all_retired = draining_;
    for (const MachineState &machine : machines_) {
      if (machine.phase != MachinePhase::kRetired) {
        all_retired = false;
        break;
      }
    }
    if (all_retired) {
      ended_ns_ = last_completed_ns_;
      complete_ = true;
      condition_.notify_all();
    }
  }

  bool wait_until_complete(double timeout_seconds) {
    if (timeout_seconds <= 0.0) {
      throw std::invalid_argument("timeout_seconds must be positive");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const bool reached = condition_.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds), [this]() {
          return complete_ || fatal_code_ != FatalCode::kNone;
        });
    if (fatal_code_ != FatalCode::kNone) {
      throw_fatal_locked();
    }
    return reached && complete_;
  }

  py::list traces() const {
    std::lock_guard<std::mutex> lock(mutex_);
    py::list result;
    for (const HopTrace &trace : traces_) {
      py::dict value = event_record_to_python(trace.event);
      value["completed_ns"] = trace.completed_ns;
      result.append(std::move(value));
    }
    return result;
  }

  py::dict inventory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return inventory_locked();
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    if (!complete_ || pending_count_locked() != 0) {
      throw std::runtime_error(
          "qualification bridge cannot close before complete zero inventory");
    }
    close_fd_locked();
    closed_ = true;
  }

  void abort_and_close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    close_fd_locked();
    closed_ = true;
    condition_.notify_all();
  }

#ifdef SGLANG_GIL_QUALIFICATION_TESTING
  void break_eventfd_for_test() {
    std::lock_guard<std::mutex> lock(mutex_);
    close_fd_locked();
  }
#endif

private:
  class DrainGuard {
  public:
    explicit DrainGuard(NativeGILQualificationBridge &bridge)
        : bridge_(bridge) {}
    ~DrainGuard() {
      std::lock_guard<std::mutex> lock(bridge_.mutex_);
      bridge_.drain_active_ = false;
    }

    DrainGuard(const DrainGuard &) = delete;
    DrainGuard &operator=(const DrainGuard &) = delete;

  private:
    NativeGILQualificationBridge &bridge_;
  };

  void enqueue_machine_locked(std::size_t machine_index) {
    MachineState &machine = machines_.at(machine_index);
    if (machine.phase != MachinePhase::kInitial &&
        machine.phase != MachinePhase::kDelivered) {
      set_fatal_locked(FatalCode::kInvalidMachinePhase, 0);
      return;
    }
    if (queue_.size() >= capacity_) {
      rejected_record_ = EventRecord{
          next_sequence_, static_cast<std::uint32_t>(machine_index),
          machine.generation_index, machine.hop_index, monotonic_raw_ns()};
      set_fatal_locked(FatalCode::kQueueOverflow, 0);
      return;
    }
    const EventRecord record{
        next_sequence_++, static_cast<std::uint32_t>(machine_index),
        machine.generation_index, machine.hop_index, monotonic_raw_ns()};
    machine.current = record;
    machine.phase = MachinePhase::kQueued;
    queue_.push_back(record);
    signal_eventfd_locked();
  }

  void signal_eventfd_locked() {
    if (event_fd_ < 0) {
      set_fatal_locked(FatalCode::kEventfdWriteFailure, EBADF, false);
      return;
    }
    constexpr std::uint64_t increment = 1;
    for (;;) {
      const ssize_t result = ::write(event_fd_, &increment, sizeof(increment));
      if (result == static_cast<ssize_t>(sizeof(increment))) {
        ++successful_wake_count_;
        return;
      }
      if (result < 0 && errno == EINTR) {
        continue;
      }
      set_fatal_locked(FatalCode::kEventfdWriteFailure,
                       result < 0 ? errno : EIO, false);
      return;
    }
  }

  std::uint64_t consume_eventfd() {
    std::uint64_t wake_count = 0;
    for (;;) {
      int event_fd = -1;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        event_fd = event_fd_;
      }
      if (event_fd < 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        set_fatal_locked(FatalCode::kEventfdReadFailure, EBADF, false);
        return wake_count;
      }
      std::uint64_t observed = 0;
      const ssize_t result = ::read(event_fd, &observed, sizeof(observed));
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
      std::lock_guard<std::mutex> lock(mutex_);
      set_fatal_locked(FatalCode::kEventfdReadFailure,
                       result < 0 ? errno : EIO, false);
      break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    consumed_wake_count_ += wake_count;
    return wake_count;
  }

  void set_fatal_locked(FatalCode code, int system_error,
                        bool signal = true) {
    if (fatal_code_ != FatalCode::kNone) {
      return;
    }
    fatal_code_ = code;
    fatal_system_error_ = system_error;
    if (signal && event_fd_ >= 0) {
      constexpr std::uint64_t increment = 1;
      static_cast<void>(::write(event_fd_, &increment, sizeof(increment)));
    }
    condition_.notify_all();
  }

  void throw_if_fatal_locked() const {
    if (fatal_code_ != FatalCode::kNone) {
      throw_fatal_locked();
    }
  }

  [[noreturn]] void throw_fatal_locked() const {
    throw std::runtime_error(std::string("GIL qualification bridge fatal: ") +
                             fatal_code_name(fatal_code_));
  }

  static py::dict event_record_to_python(const EventRecord &record) {
    py::dict value;
    value["producer_sequence"] = record.producer_sequence;
    value["machine_index"] = record.machine_index;
    value["generation_index"] = record.generation_index;
    value["hop_index"] = record.hop_index;
    value["enqueued_ns"] = record.enqueued_ns;
    return value;
  }

  std::size_t pending_count_locked() const {
    std::size_t count = rejected_record_.has_value() ? 1 : 0;
    for (const MachineState &machine : machines_) {
      if (machine.phase == MachinePhase::kQueued ||
          machine.phase == MachinePhase::kDelivered) {
        ++count;
      }
    }
    return count;
  }

  py::dict inventory_locked() const {
    std::size_t queued_count = 0;
    std::size_t delivered_count = 0;
    std::size_t retired_count = 0;
    for (const MachineState &machine : machines_) {
      if (machine.phase == MachinePhase::kQueued) {
        ++queued_count;
      } else if (machine.phase == MachinePhase::kDelivered) {
        ++delivered_count;
      } else if (machine.phase == MachinePhase::kRetired) {
        ++retired_count;
      }
    }
    py::dict value;
    value["machine_count"] = machine_count_;
    value["hop_count"] = hop_count_;
    value["capacity"] = capacity_;
    value["queued_count"] = queued_count;
    value["delivered_unacknowledged_count"] = delivered_count;
    value["pending_count"] = pending_count_locked();
    value["retired_count"] = retired_count;
    value["transition_count"] = transition_count_;
    value["trace_count"] = traces_.size();
    value["next_sequence"] = next_sequence_;
    value["successful_wake_count"] = successful_wake_count_;
    value["consumed_wake_count"] = consumed_wake_count_;
    value["started_ns"] = started_ns_;
    value["ended_ns"] = ended_ns_;
    value["minimum_duration_ns"] = minimum_duration_ns_;
    value["minimum_transition_count"] = minimum_transition_count_;
    value["started"] = started_;
    value["draining"] = draining_;
    value["complete"] = complete_;
    value["closed"] = closed_;
    value["eventfd_open"] = event_fd_ >= 0;
    value["fatal_code"] = fatal_code_name(fatal_code_);
    value["fatal_system_error"] = fatal_system_error_;
    if (rejected_record_.has_value()) {
      value["rejected_record"] =
          event_record_to_python(rejected_record_.value());
    } else {
      value["rejected_record"] = py::none();
    }
    return value;
  }

  void close_fd_locked() noexcept {
    if (event_fd_ >= 0) {
      ::close(event_fd_);
      event_fd_ = -1;
    }
  }

  const std::size_t machine_count_;
  const std::size_t hop_count_;
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<MachineState> machines_;
  std::deque<EventRecord> queue_;
  std::vector<HopTrace> traces_;
  std::optional<EventRecord> rejected_record_;
  int event_fd_{-1};
  std::uint64_t next_sequence_{0};
  std::uint64_t transition_count_{0};
  std::uint64_t successful_wake_count_{0};
  std::uint64_t consumed_wake_count_{0};
  std::uint64_t minimum_duration_ns_{0};
  std::uint64_t minimum_transition_count_{0};
  std::uint64_t started_ns_{0};
  std::uint64_t last_completed_ns_{0};
  std::uint64_t ended_ns_{0};
  bool started_{false};
  bool draining_{false};
  bool complete_{false};
  bool closed_{false};
  bool drain_active_{false};
  FatalCode fatal_code_{FatalCode::kNone};
  int fatal_system_error_{0};
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<NativeGILQualificationBridge>(module,
                                           "GILQualificationBridge")
      .def(py::init<std::size_t, std::size_t, std::size_t>(),
           py::arg("machine_count"), py::arg("hop_count"),
           py::arg("capacity"))
      .def("fileno", &NativeGILQualificationBridge::fileno)
      .def("start", &NativeGILQualificationBridge::start,
           py::arg("minimum_duration_seconds"),
           py::arg("minimum_transition_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("drain", &NativeGILQualificationBridge::drain)
      .def("acknowledge_dispatch",
           &NativeGILQualificationBridge::acknowledge_dispatch,
           py::arg("producer_sequence"),
           py::call_guard<py::gil_scoped_release>())
      .def("wait_until_complete",
           &NativeGILQualificationBridge::wait_until_complete,
           py::arg("timeout_seconds"), py::call_guard<py::gil_scoped_release>())
      .def("traces", &NativeGILQualificationBridge::traces)
      .def("inventory", &NativeGILQualificationBridge::inventory)
      .def("close", &NativeGILQualificationBridge::close,
           py::call_guard<py::gil_scoped_release>())
      .def("abort_and_close", &NativeGILQualificationBridge::abort_and_close,
           py::call_guard<py::gil_scoped_release>())
#ifdef SGLANG_GIL_QUALIFICATION_TESTING
      .def("_break_eventfd_for_test",
           &NativeGILQualificationBridge::break_eventfd_for_test)
#endif
      ;
}
