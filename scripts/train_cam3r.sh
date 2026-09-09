#!/bin/bash
# =============================================================================
# CAM3R training: the two-phase curriculum described in the paper (Sec. 4.2).
#
#   Phase 1  CAM3R-homo   homogeneous pairs only (pano-pano, fisheye-fisheye,
#                         pinhole-pinhole). Teaches intra-projection geometry.
#                         Initialised from DUSt3R (cross-view module) and
#                         UniK3D (ray module) pre-trained weights.
#
#   Phase 2  CAM3R        heterogeneous pairs added (pano-fisheye, pano-pinhole,
#                         fisheye-pinhole) on top of the phase-1 checkpoint.
#                         This is the released model.
#
# CO3Dv2 is deliberately absent: it is held out for zero-shot evaluation only.
# MegaDepth stays pinhole-pinhole throughout, as no synthetic heterogeneity is
# injected into it.
#
# BEFORE RUNNING: every DATA_* variable below is a placeholder and must be edited
# to point at your own preprocessed copy of the dataset, as must DUST3R_CKPT.
# The script does not download or preprocess anything.
#
# Usage:
#   bash scripts/train_cam3r.sh          # phase 1 then phase 2
#   bash scripts/train_cam3r.sh 1        # phase 1 only
#   bash scripts/train_cam3r.sh 2        # phase 2 only
#
# Phase 2 resumes from ${OUTPUT_ROOT}/cam3r_homo/checkpoint-best.pth, so it
# requires a completed phase 1 (or that checkpoint supplied by other means).
# =============================================================================
set -euo pipefail

# --- paths -------------------------------------------------------------------
DATA_2D3DS_PANO="/path/to/2D3DS/preprocessed_gt_rays"
DATA_2D3DS_CROSS="/path/to/2D3DS/preprocessed_pano_fish_pin_gt_rays"
DATA_360LOC_PANO="/path/to/360Loc/preprocessed_panos_scaled"
DATA_360LOC_CROSS="/path/to/360Loc/preprocessed_panoFishPin"
DATA_ADT_FISH="/path/to/ADT/preprocessed_gt_rays"
DATA_ADT_CROSS="/path/to/ADT/preprocessed_cross_domain"
DATA_MEGADEPTH="/path/to/MegaDepth/preprocessed_gt_rays"

DUST3R_CKPT="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
OUTPUT_ROOT="checkpoints"
ADAPTER_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/dust3r/vitl.json"

# --- optimisation (supplementary, Table "Optimization hyperparameters") -------
NPROC=4                 # GPUs; effective batch = 4 GPUs x 4 batch x 2 accum = 32
BATCH_SIZE=4
ACCUM_ITER=2
BLR=0.00005             # lr = blr * effective_batch / 256
MIN_LR=1e-06
WARMUP_EPOCHS=10
EPOCHS_PHASE1=500       # the paper trains for 300-500 epochs per phase
EPOCHS_PHASE2=300
N_TRAIN=2_000           # pairs sampled per dataset per epoch
N_VAL=200

MODEL="AsymmetricCroCo3DStereo(pos_embed=\"RoPE100\", patch_embed_cls=\"ManyAR_PatchEmbed\", img_size=(512,384), head_type=\"dpt\", output_mode=\"pts3d\", depth_mode=(\"softplus\",0,inf), conf_mode=(\"exp\",1,inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter_config=\"${ADAPTER_CONFIG}\", adapter_pretrained=True, adapter_use_layernorm_pool=True)"
TRAIN_CRITERION="UniK3DCameraPolar(weight=1.0, polar_weight=3.0, polar_asym=0.7) + PoseAngularLoss(weight=1.0) + Regr3D(L21, norm_mode=\"avg_dis\")"
TEST_CRITERION="UniK3DCameraPolar(weight=1.0) + PoseAngularLoss(weight=1.0) + Regr3D(L21, norm_mode=\"avg_dis\")"

# --- dataset mixtures --------------------------------------------------------
HOMO_TRAIN="\
${N_TRAIN} @ Pano2D3DS(split=\"train\", ROOT=\"${DATA_2D3DS_PANO}\", aug_crop=16, resolution=(512,384), transform=ColorJitter) + \
${N_TRAIN} @ Pano360Loc(split=\"train\", ROOT=\"${DATA_360LOC_PANO}\", aug_crop=16, resolution=(512,384), transform=ColorJitter) + \
${N_TRAIN} @ FisheyeAria(split=\"train\", ROOT=\"${DATA_ADT_FISH}\", aug_crop=16, resolution=(512,512), transform=ColorJitter) + \
${N_TRAIN} @ MegaDepth(split=\"train\", ROOT=\"${DATA_MEGADEPTH}\", aug_crop=16, resolution=(512,384), transform=ColorJitter)"

HOMO_VAL="\
${N_VAL} @ Pano2D3DS(split=\"val\", ROOT=\"${DATA_2D3DS_PANO}\", resolution=(512,384), seed=777) + \
${N_VAL} @ Pano360Loc(split=\"val\", ROOT=\"${DATA_360LOC_PANO}\", resolution=(512,384), seed=777) + \
${N_VAL} @ FisheyeAria(split=\"val\", ROOT=\"${DATA_ADT_FISH}\", resolution=(512,512), seed=777) + \
${N_VAL} @ MegaDepth(split=\"val\", ROOT=\"${DATA_MEGADEPTH}\", resolution=(512,384), seed=777)"

HETERO_TRAIN="${HOMO_TRAIN} + \
${N_TRAIN} @ PanoFishPin2D3DS(split=\"train\", ROOT=\"${DATA_2D3DS_CROSS}\", aug_crop=16, resolution=(512,384), transform=ColorJitter) + \
${N_TRAIN} @ PanoFishPin360Loc(split=\"train\", ROOT=\"${DATA_360LOC_CROSS}\", aug_crop=16, resolution=(512,384), transform=ColorJitter) + \
${N_TRAIN} @ FisheyePinholeAria(split=\"train\", ROOT=\"${DATA_ADT_CROSS}\", aug_crop=16, resolution=(512,512), transform=ColorJitter)"

HETERO_VAL="${HOMO_VAL} + \
${N_VAL} @ PanoFishPin2D3DS(split=\"val\", ROOT=\"${DATA_2D3DS_CROSS}\", resolution=(512,384), seed=777) + \
${N_VAL} @ PanoFishPin360Loc(split=\"val\", ROOT=\"${DATA_360LOC_CROSS}\", resolution=(512,384), seed=777) + \
${N_VAL} @ FisheyePinholeAria(split=\"val\", ROOT=\"${DATA_ADT_CROSS}\", resolution=(512,512), seed=777)"

run_phase () {
  local name=$1 train=$2 val=$3 epochs=$4 pretrained=$5 extra=${6:-}
  echo "=== ${name}: starting $(date) ==="
  torchrun --nproc_per_node=${NPROC} train.py \
    --train_dataset="${train}" \
    --test_dataset="${val}" \
    --train_criterion="${TRAIN_CRITERION}" \
    --test_criterion="${TEST_CRITERION}" \
    --model="${MODEL}" \
    --pretrained="${pretrained}" ${extra} \
    --blr=${BLR} --min_lr=${MIN_LR} --warmup_epochs=${WARMUP_EPOCHS} \
    --epochs=${epochs} --batch_size=${BATCH_SIZE} --accum_iter=${ACCUM_ITER} \
    --save_freq=10 --keep_freq=10 --eval_freq=1 --print_freq=10 \
    --disable_cudnn_benchmark \
    --output_dir="${OUTPUT_ROOT}/${name}" \
    2>&1 | tee -a "${name}.log"
  echo "=== ${name}: finished $(date) ==="
}

PHASE=${1:-all}

if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
  run_phase "cam3r_homo" "${HOMO_TRAIN}" "${HOMO_VAL}" "${EPOCHS_PHASE1}" "${DUST3R_CKPT}"
fi

if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
  # Requires the phase-1 checkpoint below to exist.
  # --pretrained_drop=auto keeps the UniK3D adapter weights, which the legacy
  # prefix filter would silently discard when resuming from a phase-1 checkpoint.
  run_phase "cam3r" "${HETERO_TRAIN}" "${HETERO_VAL}" "${EPOCHS_PHASE2}" \
            "${OUTPUT_ROOT}/cam3r_homo/checkpoint-best.pth" "--pretrained_drop=auto"
fi
