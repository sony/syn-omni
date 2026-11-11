r"""
Syn-Omni × 8b-is: Quantum 64-Trit Bitplane Orthogonal Modality-Expert LoRA
(OME-LoRA-Pq) & Progressive Synergy Routing (PSR-Pq) via "Counting Divided" (p/q)
=================================================================================

Brought to you by **8b-is** (https://github.com/8b-is):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64`,
    Ternary b1.58 & `p/q` ("Counting Divided") integer dynamics
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

What This Module Adds to Sony `Syn-Omni` (OME-LoRA + PSR)
---------------------------------------------------------
In Syn-Omni (*Structured Specialization and Progressive Collaboration for
Omnimodal Embeddings*, EMNLP 2026 Findings), **Orthogonal Modality-Expert LoRA
(`OME-LoRA`)** decomposes adaptation into:
  1. A universal **`shared_lora`** path (mergeable into the frozen backbone at
     inference time via `merge_shared_and_unload`), and
  2. $M$ modality-expert LoRA paths (`MixtureOfLoRA` over Image, Video, Audio,
     and AudioVisual experts) regularized by an $A$-matrix orthogonality loss
     (`ortho_loss_weight`) and combined via **Progressive Synergy Routing (`PSR`)**.

This module (`ome_lora_pq.py`) introduces hardware-aligned integer arithmetic
for both OME-LoRA and PSR:

1. **Quantum 64-Trit Bitplane LoRA (`QuantumTritMatrixPq`, `14.2x` smaller)**:
   - Packs every 64-element row block of `lora_A` and `lora_B` into two 64-bit
     bitmasks `(pos_mask: u64, neg_mask: u64)` plus a `Q16` scale (`16 B` bitmasks
     per 64 weights vs `256 B` FP32; `36 B` per 128-wide row vs `512 B`, i.e.
     `14.22x`, pinned by `test_bitplane_popcnt_orthogonality_and_compression`).
2. **Bitplane `POPCNT` Orthogonality Gram Matrix & Structural Orthogonalization**:
   - The inner product between any two modality experts' $A$-matrices $A_m, A_n$
     is computed without a single floating-point multiply via bitwise `AND`/`OR`
     and hardware `POPCNT`:
       `<A_m, A_n>_trit = sum(popcnt((pos_m & pos_n) | (neg_m & neg_n))
                            - popcnt((pos_m & neg_n) | (neg_m & pos_n)))`
   - `orthogonalize_expert_bitplanes_pq` resolves colliding trits across modality
     experts so off-diagonal cosine similarity drops to ~0 by discrete bitplane
     construction.
3. **Zero-Float Progressive Synergy Router (`ProgressiveModalityRouterPq`)**:
   - Evaluates PSR routing weights using `BFP-64` (`i16 -> i64`) integer dot
     products and a rational Cayley-Padé `sigmoid_pq_q16` ($q = 2^{16}$, zero
     `expf`), smoothly interpolating between the binary modality mask $M \in \\{0,1\\}^E$
     during warmup and cross-modal synergy $\\sigma(z)$ after transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

MASK64 = (1 << 64) - 1
Q16_ONE = 1 << 16  # 65,536


def _popcnt64(x: int) -> int:
    return (x & MASK64).bit_count()


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) via math.isqrt (hardware shifts/adds)."""
    return math.isqrt(max(0, int(n)))


def sigmoid_pq_q16(x_q16: int) -> int:
    """
    Zero-float rational Padé-Cayley sigmoid in Q16:
        sigma(x) = 1 / (1 + exp(-x))  in [0, Q16_ONE]

    Uses the [2/2] Padé approximant for exp(-|x|) with exact integer arithmetic:
        exp(-u) ~= (12 - 6u + u^2) / (12 + 6u + u^2)  on u in [0, 2],
    coupled with range halving u_0 = |x| / 2^k and k squarings in Q16.
    """
    if x_q16 <= -(10 << 16):
        return 0
    if x_q16 >= (10 << 16):
        return Q16_ONE

    u = abs(x_q16)
    k = 0
    while u > Q16_ONE:
        u >>= 1
        k += 1

    # u is in [0, Q16_ONE]. Compute Padé [2/2] for exp(-u):
    # N = 12 * 1^2 - 6 * u + u^2, D = 12 * 1^2 + 6 * u + u^2
    u2_q16 = (u * u + (1 << 15)) >> 16
    twelve_q16 = 12 * Q16_ONE
    six_u_q16 = 6 * u
    num = twelve_q16 - six_u_q16 + u2_q16
    den = twelve_q16 + six_u_q16 + u2_q16
    exp_neg_abs_q16 = ((num << 16) + (den >> 1)) // den

    for _ in range(k):
        exp_neg_abs_q16 = (exp_neg_abs_q16 * exp_neg_abs_q16 + (1 << 15)) >> 16

    if x_q16 >= 0:
        # sigma(x) = 1 / (1 + exp(-x))
        denom = Q16_ONE + exp_neg_abs_q16
        return min(Q16_ONE, ((Q16_ONE << 16) + (denom >> 1)) // denom)
    else:
        # sigma(-|x|) = exp(-|x|) / (1 + exp(-|x|))
        denom = Q16_ONE + exp_neg_abs_q16
        return max(0, ((exp_neg_abs_q16 << 16) + (denom >> 1)) // denom)


@dataclass(frozen=True)
class Bfp64Block:
    """
    Omni BFP-64 Block Floating-Point vector (up to 64 lanes per block):
        x_i ~= mantissas[i] * 2^(shared_exp - 14)
    """

    shared_exp: int
    mantissas: Tuple[int, ...]

    @classmethod
    def quantize(cls, values: Sequence[float]) -> "Bfp64Block":
        max_abs = max((abs(float(v)) for v in values), default=0.0)
        if max_abs < 1e-30:
            return cls(shared_exp=-40, mantissas=tuple(0 for _ in values))
        _, exp = math.frexp(max_abs)
        shared_exp = int(exp)
        inv_scale = math.ldexp(1.0, 14 - shared_exp)
        mants = tuple(
            max(-32767, min(32767, int(round(float(v) * inv_scale))))
            for v in values
        )
        return cls(shared_exp=shared_exp, mantissas=mants)

    def dot_trit_masks(self, pos_masks: Sequence[int], neg_masks: Sequence[int]) -> int:
        """
        Compute the exact integer dot product between this BFP vector and
        64-trit bitplane masks `(pos_masks, neg_masks)` without multiplications!
        """
        acc = 0
        n = len(self.mantissas)
        for blk_idx, (pm, nm) in enumerate(zip(pos_masks, neg_masks)):
            base = blk_idx * 64
            # Fast bit-scan over active positive and negative trits
            p = pm & MASK64
            while p:
                lsb = p & -p
                bit = lsb.bit_length() - 1
                idx = base + bit
                if idx < n:
                    acc += self.mantissas[idx]
                p ^= lsb
            q = nm & MASK64
            while q:
                lsb = q & -q
                bit = lsb.bit_length() - 1
                idx = base + bit
                if idx < n:
                    acc -= self.mantissas[idx]
                q ^= lsb
        return acc


@dataclass
class QuantumTritMatrixPq:
    """
    Quantum 64-Trit Bitplane Matrix (`rows x cols`) for OME-LoRA adapters.

    Each row is stored as `ceil(cols / 64)` pairs of `(pos_mask: u64, neg_mask: u64)`
    plus a per-row `scale_q16` integer scale factor.
    """

    rows: int
    cols: int
    pos_masks: List[List[int]]
    neg_masks: List[List[int]]
    row_scales_q16: List[int]

    @property
    def blocks_per_row(self) -> int:
        return (self.cols + 63) // 64

    @property
    def byte_size(self) -> int:
        """16 bytes per 64-trit block + 4 bytes per row scale."""
        return self.rows * (self.blocks_per_row * 16 + 4)

    @property
    def fp32_byte_size(self) -> int:
        return self.rows * self.cols * 4

    @classmethod
    def from_float_matrix(cls, matrix: Sequence[Sequence[float]]) -> "QuantumTritMatrixPq":
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        blocks = (cols + 63) // 64
        pos_masks: List[List[int]] = []
        neg_masks: List[List[int]] = []
        row_scales_q16: List[int] = []

        for r in range(rows):
            row = matrix[r]
            mean_abs = sum(abs(float(v)) for v in row) / max(1, cols)
            threshold = 0.45 * mean_abs
            p_row = [0] * blocks
            n_row = [0] * blocks
            active_sum = 0.0
            active_cnt = 0
            for c, val in enumerate(row):
                fv = float(val)
                if fv > threshold:
                    p_row[c // 64] |= 1 << (c % 64)
                    active_sum += abs(fv)
                    active_cnt += 1
                elif fv < -threshold:
                    n_row[c // 64] |= 1 << (c % 64)
                    active_sum += abs(fv)
                    active_cnt += 1
            scale_f = (active_sum / active_cnt) if active_cnt > 0 else mean_abs
            row_scales_q16.append(max(1, int(round(scale_f * Q16_ONE))))
            pos_masks.append(p_row)
            neg_masks.append(n_row)

        return cls(
            rows=rows,
            cols=cols,
            pos_masks=pos_masks,
            neg_masks=neg_masks,
            row_scales_q16=row_scales_q16,
        )

    def active_trit_count(self) -> int:
        """Exact L2 norm squared of the unscaled ternary matrix via POPCNT."""
        total = 0
        for p_row, n_row in zip(self.pos_masks, self.neg_masks):
            for pm, nm in zip(p_row, n_row):
                total += _popcnt64(pm | nm)
        return total

    def inner_product_trit(self, other: "QuantumTritMatrixPq") -> int:
        """
        Exact integer Frobenius inner product `<self, other>_trit` between two
        64-trit bitplane matrices using ONLY bitwise AND/OR and POPCNT!
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix shape mismatch for bitplane inner product")
        dot = 0
        for r in range(self.rows):
            p_a, n_a = self.pos_masks[r], self.neg_masks[r]
            p_b, n_b = other.pos_masks[r], other.neg_masks[r]
            for b in range(len(p_a)):
                same_sign = (p_a[b] & p_b[b]) | (n_a[b] & n_b[b])
                opp_sign = (p_a[b] & n_b[b]) | (n_a[b] & p_b[b])
                dot += _popcnt64(same_sign) - _popcnt64(opp_sign)
        return dot

    def cosine_similarity_q15(self, other: "QuantumTritMatrixPq") -> int:
        """
        Exact integer cosine similarity in Q15 (`[-32768, 32768]`) between two
        Quantum 64-Trit matrices using POPCNT + `isqrt_u128`.
        """
        dot = self.inner_product_trit(other)
        n_a = self.active_trit_count()
        n_b = other.active_trit_count()
        if n_a == 0 or n_b == 0:
            return 0
        denom = isqrt_u128(n_a * n_b * (1 << 30))
        if denom == 0:
            return 0
        return int(((dot << 30) + (denom >> 1 if dot >= 0 else -(denom >> 1))) // denom)

    def matvec_bfp(self, x_bfp: Bfp64Block) -> List[float]:
        """
        Project input `x_bfp` (`[cols]`) through this `[rows, cols]` 64-trit
        matrix using pure integer add/sub accumulation on bitplanes.
        """
        out: List[float] = []
        for r in range(self.rows):
            acc_int = x_bfp.dot_trit_masks(self.pos_masks[r], self.neg_masks[r])
            # Reconstruct scale: x_bfp scale is 2^(shared_exp - 14), row scale is Q16
            combined = acc_int * self.row_scales_q16[r]
            out.append(math.ldexp(float(combined), x_bfp.shared_exp - 14 - 16))
        return out


def expert_orthogonality_gram_pq(
    expert_A_list: Sequence[QuantumTritMatrixPq],
) -> List[List[int]]:
    """
    Compute the `[E, E]` Q15 cosine Gram matrix across all $E$ modality experts'
    $A$-matrices using pure `POPCNT` bitplane arithmetic.

    In Syn-Omni OME-LoRA, off-diagonal entries measure subspace overlap between
    modality experts (Image, Video, Audio, AudioVisual).
    """
    num_experts = len(expert_A_list)
    gram_q15 = [[0] * num_experts for _ in range(num_experts)]
    for i in range(num_experts):
        gram_q15[i][i] = 1 << 15
        for j in range(i + 1, num_experts):
            c = expert_A_list[i].cosine_similarity_q15(expert_A_list[j])
            gram_q15[i][j] = c
            gram_q15[j][i] = c
    return gram_q15


# 64-bit Walsh-Hadamard sign-inversion masks (Z_2 x Z_2 group characters)
# across 16 nibbles of 4 trits each:
#   e=0: ++++ (flip mask 0x0000000000000000)
#   e=1: +-+- (flip mask 0xAAAAAAAAAAAAAAAA)
#   e=2: ++-- (flip mask 0xCCCCCCCCCCCCCCCC)
#   e=3: +--+ (flip mask 0x6666666666666666)
HADAMARD4_FLIP_MASKS: Tuple[int, ...] = (
    0x0000000000000000,
    0xAAAAAAAAAAAAAAAA,
    0xCCCCCCCCCCCCCCCC,
    0x6666666666666666,
)

# Disjoint sparse nibble slot masks for up to 4 modality experts
SPARSE_NIBBLE_SLOT_MASKS: Tuple[int, ...] = (
    0x1111111111111111,
    0x2222222222222222,
    0x4444444444444444,
    0x8888888888888888,
)


def orthogonalize_expert_bitplanes_pq(
    expert_A_list: Sequence[QuantumTritMatrixPq],
) -> List[QuantumTritMatrixPq]:
    """
    SWAR 64-Bit Walsh-Hadamard + Disjoint-Slot Structural Orthogonalization
    for Syn-Omni OME-LoRA Modality Experts (up to 4 experts: Image, Video,
    Audio, AudioVisual).

    Guarantees exact simultaneous mutual orthogonality `<A_m, A_n>_trit == 0`
    for ALL pairs `m != n` in O(1) branchless 64-bit bitwise operations per
    word:
      1. **Full 4-Trit Nibbles (`full_nibble_mask`)**:
         On every 4-column nibble where the union of experts has all 4 trits
         active, all 4 experts share the full 4-trit support and modulate the
         base sign pattern with the 4x4 Walsh-Hadamard group characters
         (`HADAMARD4_FLIP_MASKS`). Because `s_k^2 = 1` for active trits, the
         inner product on every full nibble is `<H_m, H_n> = 0` for all `m != n`.
      2. **Sparse Nibbles (`~full_nibble_mask`)**:
         On nibbles with fewer than 4 active trits, each expert `e` retains its
         own trit on slot `SPARSE_NIBBLE_SLOT_MASKS[e]`, giving pairwise
         disjoint support (`(sup_m & sup_n) == 0`).
    """
    cloned = [
        QuantumTritMatrixPq(
            rows=m.rows,
            cols=m.cols,
            pos_masks=[list(r) for r in m.pos_masks],
            neg_masks=[list(r) for r in m.neg_masks],
            row_scales_q16=list(m.row_scales_q16),
        )
        for m in expert_A_list
    ]
    num_experts = len(cloned)
    if num_experts <= 1:
        return cloned
    if num_experts > 4:
        raise ValueError("Walsh-Hadamard 4-nibble orthogonalizer supports up to 4 experts")

    rows = cloned[0].rows
    blocks = cloned[0].blocks_per_row

    for r in range(rows):
        for b in range(blocks):
            # Union of active support and majority/priority sign across experts
            union_pos = 0
            union_neg = 0
            for e in range(num_experts):
                pe = cloned[e].pos_masks[r][b] & MASK64
                ne = cloned[e].neg_masks[r][b] & MASK64
                already = union_pos | union_neg
                union_pos |= pe & ~already
                union_neg |= ne & ~already

            union_sup = (union_pos | union_neg) & MASK64
            # Identify 4-bit nibbles where all 4 trits are active in union_sup
            full_nibble_leaders = (
                union_sup
                & (union_sup >> 1)
                & (union_sup >> 2)
                & (union_sup >> 3)
                & 0x1111111111111111
            )
            full_nibble_mask = (full_nibble_leaders * 0xF) & MASK64
            sparse_mask = (~full_nibble_mask) & MASK64

            base_pos_full = union_pos & full_nibble_mask
            base_neg_full = union_neg & full_nibble_mask

            for e in range(num_experts):
                flip = HADAMARD4_FLIP_MASKS[e] & full_nibble_mask
                # Full nibbles: Walsh-Hadamard sign modulation of base_pos/base_neg
                p_full = (base_pos_full & ~flip) | (base_neg_full & flip)
                n_full = (base_neg_full & ~flip) | (base_pos_full & flip)

                # Sparse nibbles: disjoint expert slot mask
                slot_mask = SPARSE_NIBBLE_SLOT_MASKS[e] & sparse_mask
                p_sparse = cloned[e].pos_masks[r][b] & slot_mask
                n_sparse = cloned[e].neg_masks[r][b] & slot_mask

                cloned[e].pos_masks[r][b] = (p_full | p_sparse) & MASK64
                cloned[e].neg_masks[r][b] = (n_full | n_sparse) & MASK64

    return cloned


class ProgressiveModalityRouterPq:
    """
    Zero-Float Progressive Synergy Router (`PSR-Pq`) in `p/q` (`Q16`).

    Implements Syn-Omni's two-stage progressive routing schedule:
      1. **Warmup Phase (`progress_q16 <= warmup_q16`)**:
         Hard modality specialization using the binary modality mask
         `modality_mask in {0, 1}^num_experts` (experts first establish clean
         modality-specific priors).
      2. **Transition & Synergy Phase (`progress_q16 > warmup_q16`)**:
         Gradually blends the hard modality prior with the learned cross-modal
         synergy gate `sigma_pq(W_router @ x)` evaluated via `BFP-64` integer
         projection and rational `sigmoid_pq_q16`.
    """

    def __init__(
        self,
        router_weights: Sequence[Sequence[float]],
        router_bias: Optional[Sequence[float]] = None,
        warmup_ratio: float = 0.25,
        transition_ratio: float = 0.50,
    ) -> None:
        self.num_experts = len(router_weights)
        self.hidden_dim = len(router_weights[0]) if self.num_experts > 0 else 0
        self.weight_blocks = [Bfp64Block.quantize(row) for row in router_weights]
        self.bias_q16 = [
            int(round(float(b) * Q16_ONE))
            for b in (router_bias if router_bias is not None else [0.0] * self.num_experts)
        ]
        self.warmup_q16 = int(round(warmup_ratio * Q16_ONE))
        self.transition_q16 = max(1, int(round(transition_ratio * Q16_ONE)))

    def get_weights_q16(
        self,
        x_pooled: Sequence[float],
        modality_mask: Sequence[int],
        progress: float = 1.0,
    ) -> List[int]:
        """
        Return `num_experts` routing weights in `Q16` (`[0, 65536]`).
        """
        progress_q16 = max(0, min(Q16_ONE, int(round(progress * Q16_ONE))))
        hard_q16 = [Q16_ONE if int(m) != 0 else 0 for m in modality_mask]

        if progress_q16 <= self.warmup_q16:
            return hard_q16

        x_bfp = Bfp64Block.quantize(x_pooled)
        soft_q16: List[int] = []
        for e in range(self.num_experts):
            w_bfp = self.weight_blocks[e]
            dot_i64 = sum(
                int(xm) * int(wm)
                for xm, wm in zip(x_bfp.mantissas, w_bfp.mantissas)
            )
            # Scale dot_i64 from 2^(exp_x + exp_w - 28) into Q16 (2^-16)
            shift = (x_bfp.shared_exp + w_bfp.shared_exp - 28) + 16
            if shift >= 0:
                logit_q16 = (dot_i64 << shift) + self.bias_q16[e]
            else:
                logit_q16 = (dot_i64 >> (-shift)) + self.bias_q16[e]
            soft_q16.append(sigmoid_pq_q16(logit_q16))

        if progress_q16 >= self.warmup_q16 + self.transition_q16:
            alpha_q16 = Q16_ONE
        else:
            alpha_q16 = ((progress_q16 - self.warmup_q16) << 16) // self.transition_q16

        # Blend: (1 - alpha) * hard + alpha * soft in Q16
        blended_q16 = [
            (((Q16_ONE - alpha_q16) * h + alpha_q16 * s + (1 << 15)) >> 16)
            for h, s in zip(hard_q16, soft_q16)
        ]
        return blended_q16


class MoELoRAwithSharedPq:
    """
    Full Syn-Omni `OME-LoRA + PSR` Layer in Quantum 64-Trit Bitplanes & `BFP-64`:
      - `shared_A`, `shared_B`: Universal semantics LoRA path
      - `expert_A[e]`, `expert_B[e]`: $E$ Orthogonal Modality-Expert LoRA paths
      - `router`: `ProgressiveModalityRouterPq`
    """

    def __init__(
        self,
        shared_A: Sequence[Sequence[float]],
        shared_B: Sequence[Sequence[float]],
        expert_A_list: Sequence[Sequence[Sequence[float]]],
        expert_B_list: Sequence[Sequence[Sequence[float]]],
        router: ProgressiveModalityRouterPq,
        shared_scaling: float = 2.0,
        expert_scaling: float = 2.0,
        enforce_structural_orthogonality: bool = True,
    ) -> None:
        # shared_A: [rank_s, in_features], shared_B: [out_features, rank_s]
        self.shared_A = QuantumTritMatrixPq.from_float_matrix(shared_A)
        self.shared_B = QuantumTritMatrixPq.from_float_matrix(shared_B)
        raw_experts_A = [QuantumTritMatrixPq.from_float_matrix(a) for a in expert_A_list]
        self.expert_A = (
            orthogonalize_expert_bitplanes_pq(raw_experts_A)
            if enforce_structural_orthogonality
            else raw_experts_A
        )
        self.expert_B = [QuantumTritMatrixPq.from_float_matrix(b) for b in expert_B_list]
        self.router = router
        self.shared_scaling = float(shared_scaling)
        self.expert_scaling = float(expert_scaling)

    def forward_delta(
        self,
        x_seq: Sequence[Sequence[float]],
        modality_mask: Sequence[int],
        progress: float = 1.0,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Compute the combined `OME-LoRA + PSR` adaptation delta for a sequence
        `x_seq` (`[seq_len, in_features]`) and return `(delta_seq, routing_weights_q16)`.
        """
        if not x_seq:
            return [], []

        # Last-token pooling for ProgressiveModalityRouter
        routing_q16 = self.router.get_weights_q16(
            x_seq[-1], modality_mask=modality_mask, progress=progress
        )

        out_seq: List[List[float]] = []
        for token_vec in x_seq:
            x_bfp = Bfp64Block.quantize(token_vec)

            # 1. Universal shared LoRA path: shared_B @ (shared_A @ x)
            h_shared = self.shared_A.matvec_bfp(x_bfp)
            h_shared_bfp = Bfp64Block.quantize(h_shared)
            delta_shared = self.shared_B.matvec_bfp(h_shared_bfp)

            # 2. Orthogonal Modality-Expert LoRA paths weighted by PSR routing_q16
            combined = [v * self.shared_scaling for v in delta_shared]
            for e, w_q16 in enumerate(routing_q16):
                if w_q16 == 0:
                    continue
                h_exp = self.expert_A[e].matvec_bfp(x_bfp)
                h_exp_bfp = Bfp64Block.quantize(h_exp)
                delta_exp = self.expert_B[e].matvec_bfp(h_exp_bfp)
                w_scale = (w_q16 / float(Q16_ONE)) * self.expert_scaling
                for o_idx, val in enumerate(delta_exp):
                    combined[o_idx] += val * w_scale

            out_seq.append(combined)

        return out_seq, routing_q16
