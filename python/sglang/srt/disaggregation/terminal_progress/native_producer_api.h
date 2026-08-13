#ifndef SGLANG_TERMINAL_PROGRESS_NATIVE_PRODUCER_API_H
#define SGLANG_TERMINAL_PROGRESS_NATIVE_PRODUCER_API_H

#include <stddef.h>
#include <stdint.h>

#define SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION 1U
#define SGLANG_TERMINAL_OWNER_PRODUCER_EVENT_SIZE 168U
#define SGLANG_TERMINAL_OWNER_PRODUCER_API_SIZE 40U

#define SGLANG_TERMINAL_OWNER_PRODUCER_FLAG_OWNER_ASSIGNED_SEQUENCE (1U << 0U)
#define SGLANG_TERMINAL_OWNER_PRODUCER_FLAG_ORDERED_RETIREMENT (1U << 1U)
#define SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS                          \
  (SGLANG_TERMINAL_OWNER_PRODUCER_FLAG_OWNER_ASSIGNED_SEQUENCE |               \
   SGLANG_TERMINAL_OWNER_PRODUCER_FLAG_ORDERED_RETIREMENT)

#define SGLANG_TERMINAL_OWNER_PRODUCER_API_CAPSULE_NAME                        \
  "sglang.terminal_owner.producer_api.v1"
#define SGLANG_TERMINAL_OWNER_PRODUCER_CONTEXT_CAPSULE_NAME                    \
  "sglang.terminal_owner.producer.v1"

#ifdef __cplusplus
#define SGLANG_TERMINAL_OWNER_NOEXCEPT noexcept
extern "C" {
#else
#define SGLANG_TERMINAL_OWNER_NOEXCEPT
#endif

typedef struct sglang_terminal_owner_producer_event_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint8_t binding_digest[32];
  uint16_t event_kind;
  uint8_t reserved_after_event_kind[6];
  uint64_t enqueued_ns;
  int32_t reason_code;
  uint8_t reserved_after_reason_code[4];
  int64_t backend_status;
  uint8_t has_receipt;
  uint8_t receipt_kind;
  uint8_t receipt_outcome;
  uint8_t reserved_before_receipt_identity[5];
  uint8_t receipt_binding_digest[32];
  uint8_t receipt_issuer_digest[32];
  uint64_t receipt_terminal_timestamp_ns;
  uint8_t receipt_nonce[16];
} sglang_terminal_owner_producer_event_v1;

typedef int (*sglang_terminal_owner_producer_submit_v1_fn)(
    void *context, const sglang_terminal_owner_producer_event_v1 *event)
    SGLANG_TERMINAL_OWNER_NOEXCEPT;

typedef int (*sglang_terminal_owner_producer_retire_v1_fn)(void *context)
    SGLANG_TERMINAL_OWNER_NOEXCEPT;

typedef int (*sglang_terminal_owner_producer_join_v1_fn)(
    void *context, uint64_t timeout_ns) SGLANG_TERMINAL_OWNER_NOEXCEPT;

typedef struct sglang_terminal_owner_producer_api_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t event_struct_size;
  uint32_t flags;
  sglang_terminal_owner_producer_submit_v1_fn submit;
  sglang_terminal_owner_producer_retire_v1_fn retire;
  sglang_terminal_owner_producer_join_v1_fn join;
} sglang_terminal_owner_producer_api_v1;

#ifdef __cplusplus
}

#include <type_traits>

static_assert(
    std::is_standard_layout_v<sglang_terminal_owner_producer_event_v1>);
static_assert(
    std::is_trivially_copyable_v<sglang_terminal_owner_producer_event_v1>);
static_assert(sizeof(sglang_terminal_owner_producer_event_v1) ==
              SGLANG_TERMINAL_OWNER_PRODUCER_EVENT_SIZE);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, abi_version) ==
              0U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, struct_size) ==
              4U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       binding_digest) == 8U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, event_kind) ==
              40U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, enqueued_ns) ==
              48U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, reason_code) ==
              56U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       backend_status) == 64U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1, has_receipt) ==
              72U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       receipt_binding_digest) == 80U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       receipt_issuer_digest) == 112U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       receipt_terminal_timestamp_ns) == 144U);
static_assert(offsetof(sglang_terminal_owner_producer_event_v1,
                       receipt_nonce) == 152U);

static_assert(std::is_standard_layout_v<sglang_terminal_owner_producer_api_v1>);
static_assert(
    std::is_trivially_copyable_v<sglang_terminal_owner_producer_api_v1>);
static_assert(sizeof(sglang_terminal_owner_producer_api_v1) ==
              SGLANG_TERMINAL_OWNER_PRODUCER_API_SIZE);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, abi_version) ==
              0U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, struct_size) ==
              4U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1,
                       event_struct_size) == 8U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, flags) == 12U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, submit) == 16U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, retire) == 24U);
static_assert(offsetof(sglang_terminal_owner_producer_api_v1, join) == 32U);
#endif

#undef SGLANG_TERMINAL_OWNER_NOEXCEPT

#endif
