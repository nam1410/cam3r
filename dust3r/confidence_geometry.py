# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Training-free pseudo confidence from 3D-3D nearest neighbour.
# For use in global alignment when per-pixel overlap confidence is not available.
#
# Intuition (same as 3D-3D correspondence matching): transform one cloud into the
# other's frame, then for each point use distance-to-nearest-neighbor in that frame
# as overlap strength. Here we do one-way NN (no mutual check) and output a soft
# confidence map exp(-min_dist/sigma) instead of discrete correspondences.
#
# Note: 1-NN confidence is inherently non-smooth (flaky): the *which* point is nearest
# can change abruptly between neighboring pixels, so min_dist (and thus conf) jumps.
# This is expected; for smoother visualization apply a small spatial blur to conf.
# --------------------------------------------------------
import torch


def _min_dist_chunked(x, y, chunk_size=4096):
    """
    For each point in x, compute min distance to any point in y. Chunked to avoid OOM.
    x: (B, N1, 3), y: (B, N2, 3). Returns (B, N1).
    """
    B, N1, _ = x.shape
    _, N2, _ = y.shape
    device, dtype = x.device, x.dtype
    out = torch.full((B, N1), float("inf"), device=device, dtype=dtype)
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        x_chunk = x[:, i:end_i, :]
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            y_chunk = y[:, j:end_j, :]
            d = torch.cdist(x_chunk, y_chunk)
            min_d = d.min(dim=2).values
            out[:, i:end_i] = torch.minimum(out[:, i:end_i], min_d)
    return out


def _min_dist_and_argmin_chunked(x, y, chunk_size=4096):
    """
    For each point in x, min distance to y and the index in y that achieved it. Chunked.
    x: (B, N1, 3), y: (B, N2, 3). Returns (min_dist (B, N1), argmin (B, N1) int64 in [0, N2)).
    """
    B, N1, _ = x.shape
    _, N2, _ = y.shape
    device, dtype = x.device, x.dtype
    out_min = torch.full((B, N1), float("inf"), device=device, dtype=dtype)
    out_argmin = torch.zeros(B, N1, dtype=torch.long, device=device)
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        x_chunk = x[:, i:end_i, :]
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            y_chunk = y[:, j:end_j, :]
            d = torch.cdist(x_chunk, y_chunk)
            min_d = d.min(dim=2).values
            argmin_j = d.argmin(dim=2)  # (B, chunk_i) index in [0, end_j - j)
            global_argmin = j + argmin_j
            better = min_d < out_min[:, i:end_i]
            out_min[:, i:end_i] = torch.where(better, min_d, out_min[:, i:end_i])
            out_argmin[:, i:end_i] = torch.where(better, global_argmin, out_argmin[:, i:end_i])
    return out_min, out_argmin


@torch.no_grad()
def compute_overlap_confidence_from_geometry(
    pred1, pred2, sigma=0.1, scale_invariant=True, eps=1e-6, chunk_size=4096,
    valid_mask1=None, valid_mask2=None, use_mutual_nn=False,
):
    """
    Compute per-pixel overlap confidence from predicted geometry (no training).

    For each view, confidence is high where the predicted 3D point has a nearby
    counterpart in the other view (after aligning with predicted pose).

    pred1, pred2: dicts with
      - 'pts3d': (B, H, W, 3) in own camera frame
      - pred2 must have 'R' (B, 3, 3), 't' (B, 3) view2 -> view1
    sigma: scale for exp(-dist/sigma). Acts as a soft threshold: small sigma (e.g. 0.1)
      => only very close nearest-neighbor pairs get high conf (sparse bright map);
      larger sigma (e.g. 0.4–0.6) => more gradual falloff, aligned regions show higher conf.
    scale_invariant: if True, normalize distances by median scene scale before sigma.
      Then min_dist is in "multiples of median min_dist" per view, so sigma has the same
      meaning across scenes (e.g. MegaDepth ~3 units, Fisheye ~10, Panorama ~0.2): only
      relative closeness within that pair matters, not absolute units.
    valid_mask1, valid_mask2: optional (B, H1, W1) and (B, H2, W2) boolean. If provided,
      median is computed only over valid pixels; invalid pixels get conf=0. Prefer building
      masks from depth (e.g. exclude pixels in the bottom percentile of the depth distribution)
      so padding/near-camera artifacts are excluded without masking dark objects (e.g. black chair).
    use_mutual_nn: if True, only pixels that are mutual nearest-neighbors get non-zero confidence
      (stricter, sparser map; can reduce one-way NN flakiness and false high conf).
    eps: small constant to avoid div by zero.
    chunk_size: max points per chunk for cdist (avoids OOM on large H*W; default 4096).

    Returns:
      conf1: (B, H1, W1) confidence for view1
      conf2: (B, H2, W2) confidence for view2
    """
    pts1 = pred1["pts3d"]  # (B, H1, W1, 3)
    pts2 = pred2["pts3d"]  # (B, H2, W2, 3)
    R = pred2["R"]         # (B, 3, 3) or (3, 3) view2 -> view1
    t = pred2["t"]         # (B, 3) or (3,)

    # Ensure batched: (B, 3, 3) and (B, 3)
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    B, H1, W1, _ = pts1.shape
    _, H2, W2, _ = pts2.shape

    # View2 points in view1 frame: batched (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    pts2_flat = pts2.reshape(B, -1, 3)  # (B, N2, 3)
    pts2_in_v1 = torch.einsum("bij,bnj->bni", R, pts2_flat) + t.unsqueeze(1)
    pts2_in_v1 = pts2_in_v1.reshape(B, H2, W2, 3)

    # View1 points in view2 frame: R_1to2 = R^T, t_1to2 = -R^T @ t
    R_1to2 = R.transpose(-2, -1)
    t_1to2 = -torch.einsum("bij,bj->bi", R_1to2, t)
    pts1_flat = pts1.reshape(B, -1, 3)
    pts1_in_v2 = torch.einsum("bij,bnj->bni", R_1to2, pts1_flat) + t_1to2.unsqueeze(1)
    pts1_in_v2 = pts1_in_v2.reshape(B, H1, W1, 3)

    pts1_flat = pts1.reshape(B, -1, 3)
    pts2_flat = pts2_in_v1.reshape(B, -1, 3)
    N1, N2 = pts1_flat.shape[1], pts2_flat.shape[1]
    if use_mutual_nn:
        min_dist1, idx1_to_2 = _min_dist_and_argmin_chunked(pts1_flat, pts2_flat, chunk_size=chunk_size)
        min_dist2, idx2_to_1 = _min_dist_and_argmin_chunked(pts2_flat, pts1_flat, chunk_size=chunk_size)
    else:
        min_dist1 = _min_dist_chunked(pts1_flat, pts2_flat, chunk_size=chunk_size)
        min_dist2 = _min_dist_chunked(pts2_flat, pts1_flat, chunk_size=chunk_size)
    min_dist1 = min_dist1.reshape(B, H1, W1)
    min_dist2 = min_dist2.reshape(B, H2, W2)

    if use_mutual_nn:
        idx1_to_2 = idx1_to_2.reshape(B, H1, W1)
        idx2_to_1 = idx2_to_1.reshape(B, H2, W2)
        mutual1 = torch.zeros(B, H1, W1, dtype=torch.bool, device=pts1.device)
        mutual2 = torch.zeros(B, H2, W2, dtype=torch.bool, device=pts2.device)
        for b in range(B):
            # view1 point i's nearest in view2 is j = idx1_to_2[b,i]; view2 point j's nearest in view1 is idx2_to_1[b,j]. Mutual iff idx2_to_1[b, j] == i (flat index).
            idx1_flat = idx1_to_2[b].view(-1)  # (H1*W1,)
            back = idx2_to_1[b].view(-1)[idx1_flat]  # (H1*W1,) view1 flat index that view2's match thinks is nearest
            mutual1[b] = (back == torch.arange(N1, device=pts1.device, dtype=torch.long)).view(H1, W1)
            idx2_flat = idx2_to_1[b].view(-1)
            back2 = idx1_to_2[b].view(-1)[idx2_flat]
            mutual2[b] = (back2 == torch.arange(N2, device=pts2.device, dtype=torch.long)).view(H2, W2)

    # Scale-invariant: express distance in multiples of median (per view) so sigma
    # is comparable across datasets. If valid masks given, use median over valid pixels only
    # (e.g. exclude black padding so panorama scene scale is not corrupted by padding).
    if scale_invariant:
        if valid_mask1 is not None:
            med1 = torch.zeros(B, 1, device=min_dist1.device, dtype=min_dist1.dtype)
            for b in range(B):
                v = valid_mask1[b]
                if v.any():
                    med1[b] = min_dist1[b][v].median().clamp(min=eps)
                else:
                    med1[b] = min_dist1[b].median().clamp(min=eps)
        else:
            med1 = min_dist1.view(B, -1).median(dim=1, keepdim=True).values.clamp(min=eps)
        if valid_mask2 is not None:
            med2 = torch.zeros(B, 1, device=min_dist2.device, dtype=min_dist2.dtype)
            for b in range(B):
                v = valid_mask2[b]
                if v.any():
                    med2[b] = min_dist2[b][v].median().clamp(min=eps)
                else:
                    med2[b] = min_dist2[b].median().clamp(min=eps)
        else:
            med2 = min_dist2.view(B, -1).median(dim=1, keepdim=True).values.clamp(min=eps)
        min_dist1 = min_dist1 / med1.view(B, 1, 1)
        min_dist2 = min_dist2 / med2.view(B, 1, 1)

    conf1 = torch.exp(-min_dist1 / sigma)
    conf2 = torch.exp(-min_dist2 / sigma)

    if use_mutual_nn:
        conf1 = conf1.where(mutual1, torch.zeros_like(conf1))
        conf2 = conf2.where(mutual2, torch.zeros_like(conf2))
    if valid_mask1 is not None:
        conf1 = conf1.where(valid_mask1, torch.zeros_like(conf1))
    if valid_mask2 is not None:
        conf2 = conf2.where(valid_mask2, torch.zeros_like(conf2))

    return conf1, conf2


def apply_geometry_confidence_to_predictions(pred1, pred2, sigma=0.1, scale_invariant=True):
    """
    Overwrite pred1['conf'] and pred2['conf'] with geometry-based pseudo confidence.
    Call before global_aligner so alignment uses overlap-based weighting.
    """
    conf1, conf2 = compute_overlap_confidence_from_geometry(
        pred1, pred2, sigma=sigma, scale_invariant=scale_invariant
    )
    # Global alignment may expect conf in a certain range (e.g. log-scale or exp).
    # Keep conf in (0, 1] so that conf_trf and min_conf_thr behave; optionally clip min.
    pred1["conf"] = conf1.clamp(min=1e-6)
    pred2["conf"] = conf2.clamp(min=1e-6)
    return pred1, pred2


@torch.no_grad()
def compute_pairwise_mutual_nn_confidence(
    pred1, pred2, eps=1e-6, chunk_size=4096
):
    """
    Compute strict Mutual Nearest Neighbour (MNN) distances between two views.

    - Transform pred2['pts3d'] into pred1's frame using pred2['R'], pred2['t'].
    - Find NN indices + distances for view1->view2 and view2->view1 (in pred1 frame).
    - A pair (i,j) is mutual iff NN_1to2[i] == j and NN_2to1[j] == i.
    - For mutual pairs, store the raw Euclidean 3D distance (same value in both views).
    - Non-mutual pixels get NaN.

    Args:
      pred1, pred2: dicts with
        - 'pts3d': (B, H, W, 3) in their own camera frames
        - pred2 must have 'R' (B,3,3), 't' (B,3) view2 -> view1

    Returns:
      mnn_dist1: (B, H1, W1) distances for view1 (NaN for non-mutual)
      mnn_dist2: (B, H2, W2) distances for view2 (NaN for non-mutual)
      mnn_mask1: (B, H1, W1) bool mask of mutual matches in view1
      mnn_mask2: (B, H2, W2) bool mask of mutual matches in view2
    """
    pts1 = pred1["pts3d"]  # (B, H1, W1, 3)
    pts2 = pred2["pts3d"]  # (B, H2, W2, 3)
    R = pred2["R"]         # (B, 3, 3) or (3,3) view2 -> view1
    t = pred2["t"]         # (B, 3)

    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    B, H1, W1, _ = pts1.shape
    _, H2, W2, _ = pts2.shape

    # View2 points in view1 frame
    pts2_flat = pts2.reshape(B, -1, 3)  # (B, N2, 3)
    pts2_in_v1 = torch.einsum("bij,bnj->bni", R, pts2_flat) + t.unsqueeze(1)
    pts2_in_v1 = pts2_in_v1.reshape(B, H2, W2, 3)

    # Flatten both clouds in view1 frame
    pts1_flat = pts1.reshape(B, -1, 3)                # (B, N1, 3) in view1 frame
    pts2_flat_v1 = pts2_in_v1.reshape(B, -1, 3)       # (B, N2, 3) in view1 frame
    N1, N2 = pts1_flat.shape[1], pts2_flat_v1.shape[1]

    # Forward and backward NN with argmin
    d1, idx1_to_2 = _min_dist_and_argmin_chunked(pts1_flat, pts2_flat_v1, chunk_size=chunk_size)
    d2, idx2_to_1 = _min_dist_and_argmin_chunked(pts2_flat_v1, pts1_flat, chunk_size=chunk_size)
    # Initialize outputs with NaN
    nan = float("nan")
    mnn_dist1 = torch.full((B, N1), nan, device=pts1.device, dtype=pts1.dtype)
    mnn_dist2 = torch.full((B, N2), nan, device=pts2.device, dtype=pts1.dtype)
    mnn_mask1 = torch.zeros(B, N1, dtype=torch.bool, device=pts1.device)
    mnn_mask2 = torch.zeros(B, N2, dtype=torch.bool, device=pts2.device)

    for b in range(B):
        idx1 = idx1_to_2[b]  # (N1,) in 0..N2-1
        idx2 = idx2_to_1[b]  # (N2,) in 0..N1-1
        d1_b = d1[b]
        d2_b = d2[b]

        # For each view1 flat index i, view2 index j = idx1[i]; check if idx2[j] == i
        back_from2 = idx2[idx1]  # (N1,) view1 flat indices
        arange1 = torch.arange(N1, device=pts1.device, dtype=idx1.dtype)
        mutual1_flat = back_from2.eq(arange1)  # (N1,)
        mutual_i = mutual1_flat.nonzero(as_tuple=False).flatten()  # indices in view1
        if mutual_i.numel() == 0:
            continue
        mutual_j = idx1[mutual_i]  # indices in view2 (N2)

        # Symmetric pair distance: average of both directions
        d_pair = 0.5 * (d1_b[mutual_i] + d2_b[mutual_j])
        mnn_dist1[b, mutual_i] = d_pair
        mnn_dist2[b, mutual_j] = d_pair
        mnn_mask1[b, mutual_i] = True
        mnn_mask2[b, mutual_j] = True

    mnn_dist1 = mnn_dist1.reshape(B, H1, W1)
    mnn_dist2 = mnn_dist2.reshape(B, H2, W2)
    mnn_mask1 = mnn_mask1.reshape(B, H1, W1)
    mnn_mask2 = mnn_mask2.reshape(B, H2, W2)

    return mnn_dist1, mnn_dist2, mnn_mask1, mnn_mask2
