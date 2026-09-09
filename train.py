#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for DUSt3R
# --------------------------------------------------------
from dust3r.training_dust3r_unik3d_encoder_v2_relpose_global_scale import get_args_parser, train

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)