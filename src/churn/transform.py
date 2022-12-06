import warnings
from abc import ABC, abstractmethod

from torch import Tensor, where, ones
from numpy import ndarray


class ChurnTransform(ABC):
    @staticmethod
    def _check_softmax(y_base_model: Tensor) -> bool:
        if y_base_model.sum() == len(y_base_model):
            return True
        else:
            return False

    @staticmethod
    def _prepare_tensors(
        y_true: Tensor | ndarray, y_base_model: Tensor | ndarray
    ) -> tuple[Tensor, Tensor]:
        if isinstance(y_true, ndarray) or isinstance(y_base_model, ndarray):
            y_true = Tensor(y_true)
            y_base_model = Tensor(y_base_model)
        elif isinstance(y_true, Tensor) and isinstance(y_base_model, Tensor):
            y_true = y_true
            y_base_model = y_base_model
        else:
            raise TypeError(
                f"""y_true and y_base_model are supposed
             to be torch.Tensor or np.ndarray, got {type(y_true)} 
             and {type(y_base_model)}"""
            )
        if ChurnTransform._check_softmax(y_base_model):
            warnings.warn("Your y_base_model Tensor might be already softmaxed!")
        return y_true, y_base_model

    @abstractmethod
    def transform(
        self, y_true: Tensor | ndarray, y_base_model: Tensor | ndarray
    ) -> Tensor:
        ...


class Distillation(ChurnTransform):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def __repr__(self):
        return f"{self.__class__.__name__}(lambda={self.lambda_})"

    def transform(self, y_true, y_base_model):
        y_true, y_base_model = self._prepare_tensors(y_true, y_base_model)
        y_distill = (
            self.lambda_ * y_base_model.softmax(1) + (1.0 - self.lambda_) * y_true
        )
        return y_distill


class AnchorRCP(ChurnTransform):
    def __init__(
        self, alpha: float, epsilon: float, n_classes: int, smoothing: bool = True
    ):
        self.alpha = alpha
        self.eps = epsilon
        self.classes = n_classes
        self.smoothing = smoothing

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, eps={self.eps})"

    def transform(self, y_true, y_base_model):
        y_true, y_base_model = self._prepare_tensors(y_true, y_base_model)
        if self.smoothing:
            y_rcp = where(
                y_base_model.softmax(1).round() == y_true,
                self.alpha * y_base_model.softmax(1) + (1.0 - self.alpha) * y_true,
                self.eps
                * (
                    (1.0 - self.alpha) * y_true
                    + self.alpha / self.classes * ones((len(y_true), self.classes))
                ),
            )
        else:
            y_rcp = where(
                y_base_model.softmax(1).round() == y_true,
                self.alpha * y_base_model.softmax(1) + (1.0 - self.alpha) * y_true,
                self.eps * y_true,
            )
        return y_rcp
