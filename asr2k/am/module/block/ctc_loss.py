#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn



class CTCCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
        self.none_reduction_criterion = nn.CTCLoss(reduction='none', zero_infinity=True)

    def forward(self,
                output_tensor: torch.tensor,
                output_lengths: torch.tensor,
                target_tensor: torch.tensor,
                target_lengths: torch.tensor):

        assert torch.max(target_tensor) < output_tensor.shape[2], 'id is larger than output size'

        loss = self.criterion(output_tensor.transpose(0,1), target_tensor, output_lengths, target_lengths)
        return loss

    def non_reduction_forward(self,
                         output_tensor: torch.tensor,
                         output_lengths: torch.tensor,
                         target_tensor: torch.tensor,
                         target_lengths: torch.tensor):

        loss = self.none_reduction_criterion(output_tensor.transpose(0,1), target_tensor, output_lengths, target_lengths)
        return loss