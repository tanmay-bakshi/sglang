import zmq
from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self._file_descriptors: set[int] = set()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def register_file_descriptor(self, file_descriptor: int) -> None:
        """Register one process-lifetime fd which can wake the scheduler.

        :param file_descriptor: Open readable descriptor owned by another
            process-lifetime component.
        """

        if type(file_descriptor) is not int or file_descriptor < 0:
            raise ValueError("file_descriptor must be a non-negative integer")
        if file_descriptor in self._file_descriptors:
            raise ValueError("file_descriptor is already registered")
        self.poller.register(file_descriptor, zmq.POLLIN)
        self._file_descriptors.add(file_descriptor)

    def unregister_file_descriptor(self, file_descriptor: int) -> None:
        """Remove one fd before its owning component closes it.

        :param file_descriptor: Exact descriptor previously registered.
        """

        if file_descriptor not in self._file_descriptors:
            raise ValueError("file_descriptor is not registered")
        self.poller.unregister(file_descriptor)
        self._file_descriptors.remove(file_descriptor)

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()
