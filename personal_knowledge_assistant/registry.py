from typing import Callable, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")

class Registry(Generic[K, V]):
    def __init__(self, name: str):
        self.name = name
        self._factories: dict[K, Callable[..., V]] = {}
        self._default_factory: Optional[Callable[..., V]] = None


    def register(self, key: K):
        """Decorator to register a factory function for a given key."""
        def decorator(factory: Callable[..., V]) -> Callable[..., V]:
            self._factories[key] = factory
            return factory
        return decorator
    

    def default(self, factory: Callable[..., V]) -> Callable[..., V]:
        """Set a default factory function."""
        self._default_factory = factory
        return factory
    

    def create(self, key: K, *args, **kwargs) -> V:
        """Create an instance using the factory function associated with the key."""
        factory = self._factories.get(key, self._default_factory)
        if factory is None:
            raise ValueError(f"No factory registered for key '{key}' and no default factory set.")
        return factory(*args, **kwargs)
