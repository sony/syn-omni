from typing import Any, Dict, List, Tuple

import torch


def _compute_slot_token_offsets_from_grid_thw(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    grid_thw: [N_slots, 3] (T,H,W)
    Returns offsets: [N_slots+1] where slot i corresponds to rows [offsets[i]:offsets[i+1]]
    IMPORTANT: This assumes processor pixel_values rows per slot == T*H*W.
    """
    lengths = (grid_thw[:, 0].long() * grid_thw[:, 1].long() * grid_thw[:, 2].long())
    offsets = torch.zeros((lengths.numel() + 1, ), device=grid_thw.device, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return offsets


def _slice_flat_tokens_by_slot_indices(
    flat_tokens: torch.Tensor,  # [N_total, D]
    grid_thw: torch.Tensor,  # [N_slots, 3]
    slot_indices: torch.Tensor  # [K] indices into slots
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (flat_tokens_sliced, grid_thw_sliced) for selected slots, preserving slot order.
    """
    if slot_indices.numel() == 0:
        # Return empty tensors with correct shapes/dtypes
        empty_tokens = flat_tokens.new_zeros((0, flat_tokens.shape[1]))
        empty_grid = grid_thw.new_zeros((0, 3))
        return empty_tokens, empty_grid

    # ensure sorted to preserve "processor order" within the selected subset
    slot_indices = torch.sort(slot_indices).values

    offsets = _compute_slot_token_offsets_from_grid_thw(grid_thw)  # [N_slots+1]
    pieces = []
    for i in slot_indices.tolist():
        s = offsets[i].item()
        e = offsets[i + 1].item()
        pieces.append(flat_tokens[s:e])
    return torch.cat(pieces, dim=0), grid_thw.index_select(0, slot_indices)


def _slice_video_second_per_grid(
    video_second_per_grid: torch.Tensor,  # [N_vid_slots]
    slot_indices: torch.Tensor
) -> torch.Tensor:
    if slot_indices.numel() == 0:
        return video_second_per_grid.new_zeros((0, ))
    slot_indices = torch.sort(slot_indices).values
    return video_second_per_grid.index_select(0, slot_indices)


def _slice_audio_by_slot_indices(
    input_features: torch.Tensor,  # [N_aud, 128, T] (example)
    feature_attention_mask: torch.Tensor,  # [N_aud, T]
    slot_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if slot_indices.numel() == 0:
        return (
            input_features.new_zeros((0, ) + tuple(input_features.shape[1:])),
            feature_attention_mask.new_zeros((0, ) + tuple(feature_attention_mask.shape[1:])),
        )
    slot_indices = torch.sort(slot_indices).values
    return (
        input_features.index_select(0, slot_indices),
        feature_attention_mask.index_select(0, slot_indices),
    )


def split_dense_inputs(
    model_input: Any,
    chunk_size: int,
) -> List[Dict[str, Dict[str, torch.Tensor]]]:
    """
    model_input comes from your trainer as ({"query": queries_dict}, queries_mm) or
    ({"passage": passages_dict}, passages_mm).

    We will:
      - split text batch tensors by batch dimension
      - for each chunk, slice multimodal slots using mm mapping dict
      - reconstruct per-chunk dict matching Qwen Omni forward signature

    Returns: a list of chunks, with each chunk in dict looks like
       [{"query": chunk_inputs_dict}, ...] or [{"passage": chunk_inputs_dict}, ...]
    """
    assert isinstance(model_input, (tuple, list)) and len(model_input) == 2, type(model_input)
    packed, mm = model_input
    assert isinstance(packed, dict) and len(packed) == 1
    arg_key = next(iter(packed.keys()))  # "query" or "passage"
    full = packed[arg_key]  # dict of tensors (processor output)
    assert isinstance(mm, dict), "mm must be the mapping dict from collator"

    # batch size inferred from input_ids
    bsz = full["input_ids"].shape[0]

    # create chunk index ranges: [0:chunk_size], [chunk_size:2*chunk_size], ...
    chunks = []
    for start in range(0, bsz, chunk_size):
        end = min(start + chunk_size, bsz)
        batch_idx = torch.arange(start, end, device=full["input_ids"].device, dtype=torch.long)

        # 1) slice the standard batch-first tensors by batch_idx
        chunk_full: Dict[str, torch.Tensor] = {}
        for k, v in full.items():
            # split text: 'input_ids' and 'attention_mask' as chunks
            if k in ["input_ids", "attention_mask", "modality_mask"]:
                assert isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == bsz
                chunk_full[k] = v.index_select(0, batch_idx)

        # 2) slice image slots (if present)
        if "image_grid_thw" in full and "pixel_values" in full:
            img_batch_idx = mm.get("image_batch_idx", None)
            assert img_batch_idx is not None

            # slot indices whose owner batch is in this chunk
            img_slot_mask = (img_batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)).any(dim=0)
            img_slots = torch.nonzero(img_slot_mask, as_tuple=False).flatten().to(torch.long)

            pv, gthw = _slice_flat_tokens_by_slot_indices(
                full["pixel_values"], full["image_grid_thw"], img_slots
            )
            chunk_full["pixel_values"] = pv
            chunk_full["image_grid_thw"] = gthw

        # 3) slice video slots (if present)
        if "video_grid_thw" in full and "pixel_values_videos" in full:
            vid_batch_idx = mm.get("video_batch_idx", None)
            assert vid_batch_idx is not None

            vid_slot_mask = (vid_batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)).any(dim=0)
            vid_slots = torch.nonzero(vid_slot_mask, as_tuple=False).flatten().to(torch.long)

            pvv, vgthw = _slice_flat_tokens_by_slot_indices(
                full["pixel_values_videos"], full["video_grid_thw"], vid_slots
            )
            chunk_full["pixel_values_videos"] = pvv
            chunk_full["video_grid_thw"] = vgthw

            if "video_second_per_grid" in full:
                chunk_full["video_second_per_grid"] = _slice_video_second_per_grid(
                    full["video_second_per_grid"], vid_slots
                )

        # 4) slice audio slots (if present)
        if "input_features" in full and "feature_attention_mask" in full:
            aud_batch_idx = mm.get("audio_batch_idx", None)
            assert aud_batch_idx is not None

            aud_slot_mask = (aud_batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)).any(dim=0)
            aud_slots = torch.nonzero(aud_slot_mask, as_tuple=False).flatten().to(torch.long)

            feats, fam = _slice_audio_by_slot_indices(
                full["input_features"], full["feature_attention_mask"], aud_slots
            )
            chunk_full["input_features"] = feats
            chunk_full["feature_attention_mask"] = fam

        # Drop empty tensors to avoid downstream issues (preferred over passing empty tensors)
        # This implements: "if v is empty tensor: remove k from chunk_full dict"
        keys_to_remove = []
        for k, v in chunk_full.items():
            if torch.is_tensor(v) and v.numel() == 0:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            chunk_full.pop(k, None)

        chunks.append({arg_key: chunk_full})

    return chunks
