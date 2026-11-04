# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main():
    return get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        print(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        print(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if is_main():
        print(message)
