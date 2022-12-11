"""
Contains a base class for churn transformation and knowledge distiallation
and anchor rcp methods for prediction churn reduction.
"""

import warnings
from abc import ABC, abstractmethod

from torch import Tensor, ones, concat, gather, all as tall
from numpy import ndarray


class ChurnTransform(ABC):
    """
    Base class for all churn reduction transformations.
    """

    @staticmethod
    def _check_softmax(y_base_model: Tensor) -> bool:
        """
        Helper method for raising a warning if a tensor adds up to 1.
        :param y_base_model: teacher/base model prediction tensor
        :return: True if tensor sums up to 1, False otherwise
        """
        if y_base_model.sum() == len(y_base_model):
            return True
        else:
            return False

    @staticmethod
    def _prepare_tensors(
        y_true: Tensor | ndarray, y_base_model: Tensor | ndarray
    ) -> tuple[Tensor, Tensor]:
        """
        Helper method for working with numpy ndarray and torch Tensor types.
        Raise TypeError if y_true/y_base_model are not np.ndarray or torch.Tensor
        types.
        Warns if y_base_model sums up to 1.
        :param y_true: true labels
        :param y_base_model: teacher/base model predictions, not softmaxed
        :return: tuple of torch Tensors
        """
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
        """
        Abstract method for implementing in all churn transformation subclasses
        :param y_true: true labels
        :param y_base_model: teacher/base model predictions
        :return: transformed torch Tensor
        """
        ...


class Distillation(ChurnTransform):
    """
    Knowledge distillation transformation.
    Based on https://arxiv.org/pdf/2106.02654.pdf paper.
    The method has one hyperparameter lambda_ and labels transformation
    are in the form:
    y_transformed = lambda * y_teacher + (1 - lambda) * y_true
    """

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
    """
    Anchor RCP (Regress to Correct Prediction).
    Based on https://papers.nips.cc/paper/2016/file/dc5c768b5dc76a084531934b34601977-Paper.pdf paper.
    The method has two hyperparameters alpha and epsilon. Original transformation
    equation:

    \hat{y} =
    \begin{cases}
    \alpha\times y_{teacher} + (1 -\alpha) \times y_{true}, \text{when } argmax(y_{teacher}) = argmax(y_{true}) \\
    \epsilon \times y_{true}, \text{otherwise} \\
    \end{cases}

    If smoothing parameter is on then the transformation is the following:

    \hat{y} =
    \begin{cases}
    \alpha\times y_{teacher} + (1 -\alpha) \times y_{true}, \text{when } argmax(y_{teacher}) = argmax(y_{true}) \\
    \epsilon \times ((1 - \alpha) \times y_{true} + \frac{\alpha}{d}\times \mathbb{1}), \text{where } d \text{ is number of classes and } \mathbb{1} \text{ is a sum vector} \\
    \end{cases}
    """

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

        # mask is used for separating cases when y_true == y_base_model
        mask = tall(y_true == y_base_model.softmax(1).round(), axis=1)
        combination = (
            self.alpha * y_base_model.softmax(1)[mask]
            + (1.0 - self.alpha) * y_true[mask]
        )

        if self.smoothing:
            scaling = self.eps * (
                1.0 - self.alpha
            ) * y_true + self.alpha / self.classes * ones((len(y_true), self.classes))
        else:
            scaling = self.eps * y_true[~mask]

        y_rcp = concat([combination, scaling])

        # indices to gather tensor in original sorting
        indices = concat(
            [
                mask.nonzero().repeat(1, self.classes),
                (mask == False).nonzero().repeat(1, self.classes),
            ]
        )
        return gather(y_rcp, 0, indices.argsort(dim=0))
