from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class ModelProcessor(ABC):
    processor: ClassVar[str | None]

    @property
    def model(self) -> str | None:
        return None

    @abstractmethod
    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float] | np.ndarray | None:
        pass
