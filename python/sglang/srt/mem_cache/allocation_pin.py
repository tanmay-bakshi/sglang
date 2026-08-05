import dataclasses
import threading
from typing import Protocol

import torch

_PIN_CONSTRUCTION_SEAL = object()


class AllocationPinnedError(RuntimeError):
    """An allocator operation would invalidate an active allocation pin."""


class AllocationPin:
    """Opaque claim on exact allocator-owned page identities."""

    __slots__ = ("_registry_nonce", "_token")

    _registry_nonce: object
    _token: object

    def __init__(
        self,
        registry_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct an allocator-owned pin.

        :param registry_nonce: Exact issuing registry identity.
        :param token: Registry-private record key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _PIN_CONSTRUCTION_SEAL:
            raise TypeError("allocation pins are registry owned")
        self._registry_nonce = registry_nonce
        self._token = token


@dataclasses.dataclass(frozen=True)
class AllocationPinSnapshot:
    """Immutable virtual and physical page identities held by one pin.

    :ivar allocator_label: Diagnostic allocator identity.
    :ivar page_size: Tokens represented by one page.
    :ivar virtual_pages: Canonically ordered allocator-visible page IDs.
    :ivar physical_pages: Corresponding immutable physical page IDs.
    :ivar quarantined: Whether reuse is prohibited for the process lifetime.
    """

    allocator_label: str
    page_size: int
    virtual_pages: tuple[int, ...]
    physical_pages: tuple[int, ...]
    quarantined: bool


@dataclasses.dataclass(frozen=True)
class RequestSlotPinSnapshot:
    """Immutable request-pool slot identity and reuse generation.

    :ivar pool_label: Diagnostic request-pool identity.
    :ivar slot: Exact request-pool slot.
    :ivar generation: Allocator-derived slot reuse generation.
    :ivar quarantined: Whether reuse is prohibited for the process lifetime.
    """

    pool_label: str
    slot: int
    generation: int
    quarantined: bool


@dataclasses.dataclass
class _AllocationPinRecord:
    """Private mutable ownership record for one allocation pin."""

    pin: AllocationPin
    owner: object
    page_ids: tuple[int, ...]
    quarantined: bool = False


class AllocationPinRegistry:
    """Exact-object owner preventing page reuse while asynchronous work exists."""

    _allocator_label: str
    _lock: threading.Lock
    _owners_by_page: dict[int, object]
    _records: dict[object, _AllocationPinRecord]
    _registry_nonce: object

    def __init__(self, allocator_label: str) -> None:
        """Initialize an empty allocator-local registry.

        :param allocator_label: Stable diagnostic allocator identity.
        """

        if len(allocator_label) == 0:
            raise ValueError("allocator_label must not be empty")
        self._allocator_label = allocator_label
        self._lock = threading.Lock()
        self._owners_by_page = {}
        self._records = {}
        self._registry_nonce = object()

    def acquire(
        self,
        page_ids: tuple[int, ...],
        owner: object,
    ) -> AllocationPin:
        """Pin exact positive page IDs for one authority owner.

        :param page_ids: Canonically ordered unique page identities.
        :param owner: Exact authority allowed to release the pin.
        :returns: Opaque allocator-owned pin.
        :raises AllocationPinnedError: If any page is already pinned.
        """

        if owner is None:
            raise ValueError("allocation pin owner must not be None")
        owned_page_ids = tuple(page_ids)
        if len(owned_page_ids) == 0:
            raise ValueError("allocation pin must contain at least one page")
        if owned_page_ids != tuple(sorted(set(owned_page_ids))):
            raise ValueError("allocation pin pages must be sorted and unique")
        if owned_page_ids[0] <= 0:
            raise ValueError("allocation pin pages must exclude reserved page zero")

        with self._lock:
            conflicts = tuple(
                page_id for page_id in owned_page_ids if page_id in self._owners_by_page
            )
            if len(conflicts) > 0:
                raise AllocationPinnedError(
                    f"{self._allocator_label} pages are already pinned: {conflicts}"
                )
            token = object()
            pin = AllocationPin(
                self._registry_nonce,
                token,
                _PIN_CONSTRUCTION_SEAL,
            )
            record = _AllocationPinRecord(
                pin=pin,
                owner=owner,
                page_ids=owned_page_ids,
            )
            self._records[token] = record
            for page_id in owned_page_ids:
                self._owners_by_page[page_id] = token
            return pin

    def page_ids(self, pin: AllocationPin) -> tuple[int, ...]:
        """Return immutable page IDs for an exact live pin.

        :param pin: Exact registry-owned pin.
        :returns: Canonically ordered page identities.
        """

        with self._lock:
            return self._validate_locked(pin).page_ids

    def is_quarantined(self, pin: AllocationPin) -> bool:
        """Return whether one exact pin is permanently retained.

        :param pin: Exact registry-owned pin.
        :returns: Whether the pin is quarantined.
        """

        with self._lock:
            return self._validate_locked(pin).quarantined

    def release(self, pin: AllocationPin, owner: object) -> None:
        """Release one exact pin under its acquiring authority.

        :param pin: Exact registry-owned pin.
        :param owner: Exact acquisition owner.
        :raises AllocationPinnedError: If the pin is quarantined.
        """

        with self._lock:
            record = self._validate_owner_locked(pin, owner)
            if record.quarantined:
                raise AllocationPinnedError(
                    "quarantined allocation pin cannot be released"
                )
            for page_id in record.page_ids:
                token = self._owners_by_page.get(page_id)
                if token is not pin._token:
                    raise RuntimeError("allocation pin ownership map is corrupt")
            for page_id in record.page_ids:
                del self._owners_by_page[page_id]
            del self._records[pin._token]

    def quarantine(self, pin: AllocationPin, owner: object) -> None:
        """Permanently prohibit reuse for one exact pin.

        :param pin: Exact registry-owned pin.
        :param owner: Exact acquisition owner.
        """

        with self._lock:
            record = self._validate_owner_locked(pin, owner)
            record.quarantined = True

    def assert_pages_reusable(self, page_ids: tuple[int, ...]) -> None:
        """Reject an operation touching any active or quarantined page.

        :param page_ids: Canonically ordered page identities to mutate or free.
        :raises AllocationPinnedError: If any page remains pinned.
        """

        with self._lock:
            conflicts = tuple(
                page_id for page_id in page_ids if page_id in self._owners_by_page
            )
            if len(conflicts) > 0:
                raise AllocationPinnedError(
                    f"{self._allocator_label} operation touches pinned pages: "
                    f"{conflicts}"
                )

    def assert_resettable(self, operation: str) -> None:
        """Reject allocator-wide mutation while any pin is live.

        :param operation: Reader-facing allocator operation.
        :raises AllocationPinnedError: If any active pin exists.
        """

        if len(operation) == 0:
            raise ValueError("operation must not be empty")
        with self._lock:
            if len(self._owners_by_page) == 0:
                return
            raise AllocationPinnedError(
                f"{self._allocator_label} cannot {operation} while "
                f"{len(self._owners_by_page)} page(s) are pinned"
            )

    def has_live_pins(self) -> bool:
        """Return whether any page remains pinned.

        :returns: Whether any active or quarantined page pin exists.
        """

        with self._lock:
            return len(self._owners_by_page) > 0

    def _validate_locked(self, pin: AllocationPin) -> _AllocationPinRecord:
        """Resolve one exact pin while the registry lock is held.

        :param pin: Candidate registry-owned pin.
        :returns: Private pin record.
        """

        if type(pin) is not AllocationPin:
            raise TypeError("pin must be AllocationPin")
        if pin._registry_nonce is not self._registry_nonce:
            raise AllocationPinnedError("allocation pin belongs to another registry")
        record = self._records.get(pin._token)
        if record is None or record.pin is not pin:
            raise AllocationPinnedError("allocation pin is not registered")
        return record

    def _validate_owner_locked(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> _AllocationPinRecord:
        """Validate exact pin and release authority under the registry lock.

        :param pin: Candidate registry-owned pin.
        :param owner: Candidate exact acquisition owner.
        :returns: Private pin record.
        """

        record = self._validate_locked(pin)
        if record.owner is not owner:
            raise AllocationPinnedError("allocation pin belongs to another owner")
        return record


class RequestSlotPinOwner:
    """Reusable request-pool mixin providing generation-bound slot pins."""

    _request_slot_pin_registry: AllocationPinRegistry
    _request_slot_pool_label: str
    free_slots: list[int]
    req_generation: torch.Tensor

    def _initialize_request_slot_pins(self, pool_label: str) -> None:
        """Initialize request-slot pin ownership before the first pool clear.

        :param pool_label: Stable diagnostic request-pool identity.
        """

        self._request_slot_pool_label = pool_label
        self._request_slot_pin_registry = AllocationPinRegistry(pool_label)

    def acquire_request_slot_pin(
        self,
        slot: int,
        expected_generation: int,
        owner: object,
    ) -> AllocationPin:
        """Pin one exact live request slot and generation.

        :param slot: Exact positive request-pool slot.
        :param expected_generation: Engine-observed allocation generation.
        :param owner: Exact authority allowed to release the pin.
        :returns: Opaque request-pool pin.
        """

        if type(slot) is not int or slot <= 0:
            raise ValueError("request slot must be a positive integer")
        if type(expected_generation) is not int or expected_generation <= 0:
            raise ValueError("request generation must be a positive integer")
        actual_generation = int(self.req_generation[slot].item())
        if actual_generation != expected_generation:
            raise AllocationPinnedError(
                "request slot generation changed before pin acquisition: "
                f"{actual_generation} != {expected_generation}"
            )
        return self._request_slot_pin_registry.acquire((slot,), owner)

    def request_slot_pin_snapshot(
        self,
        pin: AllocationPin,
    ) -> RequestSlotPinSnapshot:
        """Resolve one immutable request slot and current generation.

        :param pin: Exact request-pool pin.
        :returns: Generation-bound request slot identity.
        """

        slots = self._request_slot_pin_registry.page_ids(pin)
        if len(slots) != 1:
            raise RuntimeError("request slot pin must own exactly one slot")
        slot = slots[0]
        return RequestSlotPinSnapshot(
            pool_label=self._request_slot_pool_label,
            slot=slot,
            generation=int(self.req_generation[slot].item()),
            quarantined=self._request_slot_pin_registry.is_quarantined(pin),
        )

    def release_request_slot_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Release one exact request-slot pin.

        :param pin: Exact request-pool pin.
        :param owner: Exact acquisition authority.
        """

        self._request_slot_pin_registry.release(pin, owner)

    def quarantine_request_slot_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Permanently retain one ambiguous request slot.

        :param pin: Exact request-pool pin.
        :param owner: Exact acquisition authority.
        """

        self._request_slot_pin_registry.quarantine(pin, owner)

    def _assert_request_slot_reusable(self, slot: int) -> None:
        """Reject request-slot reuse while an allocation receipt owns it.

        :param slot: Exact request-pool slot about to be freed.
        """

        self._request_slot_pin_registry.assert_pages_reusable((slot,))

    def _assert_request_slots_resettable(self) -> None:
        """Reject pool reset while any request slot is pinned."""

        self.assert_request_slots_resettable("clear")

    def assert_request_slots_resettable(self, operation: str) -> None:
        """Reject pool-wide mutation while any request slot is pinned.

        :param operation: Reader-facing request-pool operation.
        """

        self._request_slot_pin_registry.assert_resettable(operation)

    def release_detached_request_slot(self, slot: int) -> None:
        """Return a request slot whose owning request object is detached.

        :param slot: Exact positive request-pool slot.
        """

        if type(slot) is not int or slot <= 0:
            raise ValueError("request slot must be a positive integer")
        self._assert_request_slot_reusable(slot)
        self.free_slots.append(slot)


class PinnableAllocation(Protocol):
    """Typed allocator surface consumed by allocation receipt authorities."""

    page_size: int

    def acquire_allocation_pin(
        self,
        indices: torch.Tensor,
        owner: object,
    ) -> AllocationPin:
        """Pin allocator-visible token indices.

        :param indices: Exact allocator-visible token indices.
        :param owner: Exact pin authority.
        :returns: Opaque allocator-owned pin.
        """

        ...

    def allocation_pin_snapshot(
        self,
        pin: AllocationPin,
    ) -> AllocationPinSnapshot:
        """Resolve immutable virtual and physical pages.

        :param pin: Exact allocator-owned pin.
        :returns: Immutable page mapping.
        """

        ...

    def release_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Release a pre-submission or terminal pin.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact pin authority.
        """

        ...

    def quarantine_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Permanently retain an ambiguous pin.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact pin authority.
        """

        ...

    def assert_allocation_resettable(self, operation: str) -> None:
        """Reject allocator-wide mutation while any page is pinned.

        :param operation: Reader-facing allocator operation.
        """

        ...

    def assert_allocation_indices_reusable(
        self,
        indices: torch.Tensor,
    ) -> None:
        """Reject mutation of exact pinned allocator-visible indices.

        :param indices: Exact allocator-visible token or slot IDs.
        """

        ...
