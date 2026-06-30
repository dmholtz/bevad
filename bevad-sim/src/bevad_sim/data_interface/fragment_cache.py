from collections import deque
from typing import Any, Deque, Dict


class FragmentCache:
    """Fragment cache class to cache once loaded fragments. Use it if you want to cache already loaded fragments.
    Use case: If you slice a container before data is loaded, and you access the data, it will load some fragments.
    But these fragments are not stored in the original container and
    will be loaded again if the same data is accessed from another slice.
    Caching helps e.g. to speed up interactive visualization.
    Attributes:
        max_cache_size_gb: float = 2.0 ### Max cache size in GB
        _cache_dict: Dict = {}       ### Dict that actually stores the data
        _cache_queue: Deque[str] = deque() ### Queue to keep track of order of adding items
        _cache_size:float = 0.0 ### current used memory in GB
        cache_active: bool = False ### Flag if cache is used

    Methods:
        query(name): Queries an item from the cache. Returns Null if item does not exist
        set(name, value): Add an item to the cache. Remove old items if cache is full

    """

    def __init__(self):
        super().__init__()
        self.max_cache_size_gb: float = 2.0  ### Max cache size in GB
        self._cache_dict: Dict = {}
        self._cache_queue: Deque[str] = deque()
        self._cache_size: float = 0.0
        self.cache_active: bool = False

    def _to_gb(self, nbytes):
        """Converts size in bytes to size in GB."""
        return float(nbytes) / float(1024**3)

    def query(self, name) -> Any:
        """Queries an item from the cache. Returns Null if item does not exist."""
        if not self.cache_active:
            return None
        if name in self._cache_dict:
            return self._cache_dict[name]
        return None

    def set(self, name, value):
        """Add an item to the cache. Remove old items if cache is full."""
        if not self.cache_active:
            return
        while self._cache_size + self._to_gb(value.nbytes) > self.max_cache_size_gb:
            item = self._cache_queue.popleft()
            array = self._cache_dict.pop(item)
            self._cache_size -= self._to_gb(array.nbytes)
        self._cache_dict[name] = value
        self._cache_size += self._to_gb(value.nbytes)
        self._cache_queue.append(name)


## the global cache instance.
fragment_cache = FragmentCache()
