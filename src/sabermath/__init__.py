from .schemas import Branch, BranchResult, Report, Task, TaskResult

_LAZY = {
    "evaluate": ".benchmark",
    "EmbeddingProcessor": ".processors",
}

__all__ = [
    "evaluate",
    "Task",
    "Report",
    "TaskResult",
    "Branch",
    "BranchResult",
    "EmbeddingProcessor",
]


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
