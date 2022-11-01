"""
This script contains functions and classes to compute churn metrics
UNTESTED
"""

import numpy as np
import torch
from typing import Union

class ChurnMetric:
    """
    Super class for metrics. 
    """
    def __init__(self, tensor_type="numpy") -> None:
        tensor_types = {"numpy": np.ndarray, "torch": torch.Tensor}
        if tensor_type not in tensor_types:
            raise NotImplementedError("Unknown object type")
        self.tensor_type = tensor_types[tensor_type]
    
    def call_sanitize_inputs(self, **preds):
        """
        preds can be true labels as well
        Shape of tensors need to be the same.
        Tensor dim len must not be >2
        """
        for p in preds:
            if not isinstance(preds[p], self.tensor_type):
                raise TypeError(f"{p} is not an instance of {str(self.tensor_type)}")
            if len(preds[p].shape) > 2:
                raise ValueError(f"Too many dims in {p}")
        shapes = set([preds[p].shape for p in preds])
        if len(shapes) > 1: #TODO extend to force only first dim to be the same
            raise ValueError(f"shape mismatch. shapes of preds must be same")
        
    def reshape_argmax(self, **preds):
        for p in preds:
            if len(preds[p].shape) == 2 and preds[p].shape[1] > 1:
                preds[p] = preds[p].argmax(1)
        return (preds[p] for p in preds)
                
class Churn(ChurnMetric):
    """
    Simple Churn. Calculates number of classification disagreements, along axis:0. 
    Will take argmax if multiple columns.
    """
    def __init__(self, tensor_type="numpy", output_mode="proportion") -> None:
        super(Churn, self).__init__(tensor_type=tensor_type)
        if output_mode not in {"proportion", "count"}: 
            raise ValueError("Unknown output_mode")
        self.output_mode = output_mode
    
    def __call__(self, predA: Union[np.ndarray, torch.Tensor], predB:Union[np.ndarray, torch.Tensor]) -> None:
        self.call_sanitize_inputs(predA=predA, predB=predB)
        predA, predB = self.reshape_argmax(predA=predA, predB=predB)
        
        churn = sum(predA!=predB)

        if self.output_mode == "proportion":
            return churn / predA.shape[0]
        if self.output_mode == "count":
            return churn

class WinLossRatio(ChurnMetric):
    """
    Lateral Churns are not loss
    """
    def __init__(self, tensor_type="numpy") -> None:
        super().__init__(tensor_type)
    
    def __call__(self, true_labels, pred_teacher, pred_student):
        self.call_sanitize_inputs(true_labels=true_labels, pred_teacher=pred_teacher, pred_student=pred_student)
        true_labels, pred_teacher, pred_student = self.reshape_argmax(true_labels, pred_teacher, pred_student)
        
        pred_teacher = pred_teacher == true_labels
        pred_student = pred_student == true_labels
        wins = sum(pred_student > pred_teacher)
        losses = sum(pred_student < pred_teacher)

        return wins / (losses + 1e-7)

class ChurnRatio(ChurnMetric):
    def __init__(self, tensor_type="numpy") -> None:
        super().__init__(tensor_type)
    
    def __call__(self, pred_teacher, pred_student, pred_control):
        self.call_sanitize_inputs(pred_teacher=pred_teacher, pred_student=pred_student, pred_control=pred_control)
        pred_teacher, pred_student, pred_control = self.reshape_argmax(pred_teacher, pred_student, pred_control)

        churnratio = sum(pred_student!=pred_teacher) / sum(pred_control!=pred_teacher)
        return churnratio

class GoodBadChurn(ChurnMetric):
    """
    lateral churn is bad
    """
    def __init__(self, tensor_type="numpy", mode=None, output_mode="proportion") -> None:
        super().__init__(tensor_type)
        if mode is None or mode not in {"good", "bad"}:
            raise ValueError("Please specify mode as good or bad")
        self.mode = mode
        if output_mode not in {"proportion", "count"}:
            raise ValueError("Unknown output_mode")
        self.output_mode = output_mode

    def __call__(self, true_labels, pred_teacher, pred_student):
        self.call_sanitize_inputs(true_labels=true_labels, pred_teacher=pred_teacher, pred_student=pred_student)
        true_labels, pred_teacher, pred_student = self.reshape_argmax(true_labels, pred_teacher, pred_student)
     
        if self.mode == "good":
            churn = sum(
                (pred_student == true_labels) > (pred_teacher == true_labels)
            )
            if self.output_mode == "proportion":
                denominator = sum((pred_student == true_labels))
                return churn / denominator
            elif self.output_mode == "count":
                return churn

        elif self.mode == "bad":
            churn = sum(
                (pred_student == true_labels) < (pred_teacher == true_labels) 
            ) + sum(
                (pred_student != true_labels) & (pred_teacher != true_labels) & (pred_teacher != pred_student)
            )
            if self.output_mode == "proportion":
                raise NotImplementedError("IDK")
            elif self.output_mode == "count":
                return churn