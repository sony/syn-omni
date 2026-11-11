import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vlm2vec_eval", "src", "model", "lora")
    ),
)

from ome_lora_pq import (
    MoELoRAwithSharedPq,
    ProgressiveModalityRouterPq,
    Q16_ONE,
    QuantumTritMatrixPq,
    expert_orthogonality_gram_pq,
    orthogonalize_expert_bitplanes_pq,
    sigmoid_pq_q16,
)


class TestSynOmniOmeLoraPq(unittest.TestCase):
    def test_rational_sigmoid_pq_q16_fidelity(self):
        for i in range(-80, 81):
            x = i * 0.1
            x_q16 = int(round(x * Q16_ONE))
            approx = sigmoid_pq_q16(x_q16) / float(Q16_ONE)
            exact = 1.0 / (1.0 + math.exp(-x))
            self.assertAlmostEqual(approx, exact, delta=1.5e-3)

    def test_bitplane_popcnt_orthogonality_and_compression(self):
        in_features = 128
        rank = 8
        num_experts = 4  # Image, Video, Audio, AudioVisual

        # Deliberately correlated initial expert A matrices
        raw_experts = []
        for e in range(num_experts):
            mat = [
                [
                    math.sin(0.11 * (r + 1) * (c + 1)) + 0.65 * math.cos(0.07 * (c + e))
                    for c in range(in_features)
                ]
                for r in range(rank)
            ]
            raw_experts.append(QuantumTritMatrixPq.from_float_matrix(mat))

        # Check compression ratio: 128 cols = 2 blocks of 64 trits (32B) + 4B scale = 36B/row
        # vs 128 * 4 = 512B/row FP32 -> 14.22x compression
        self.assertGreater(raw_experts[0].fp32_byte_size / raw_experts[0].byte_size, 14.0)

        ortho_experts = orthogonalize_expert_bitplanes_pq(raw_experts)
        gram_after = expert_orthogonality_gram_pq(ortho_experts)
        for i in range(num_experts):
            self.assertEqual(gram_after[i][i], 1 << 15)
            for j in range(i + 1, num_experts):
                self.assertEqual(
                    ortho_experts[i].inner_product_trit(ortho_experts[j]),
                    0,
                    msg=f"Experts {i} and {j} should have exact 0 bitplane inner product",
                )
                self.assertEqual(gram_after[i][j], 0)

    def test_progressive_synergy_routing_and_ome_lora_forward(self):
        in_features = 64
        out_features = 64
        shared_rank = 8
        expert_rank = 4
        num_experts = 4

        shared_A = [
            [math.sin(0.09 * (r + 1) * (c + 1)) for c in range(in_features)]
            for r in range(shared_rank)
        ]
        shared_B = [
            [math.cos(0.13 * (o + 1) * (r + 1)) for r in range(shared_rank)]
            for o in range(out_features)
        ]
        expert_A_list = [
            [
                [math.sin(0.17 * (e + 1) * (r + 1) + 0.05 * c) for c in range(in_features)]
                for r in range(expert_rank)
            ]
            for e in range(num_experts)
        ]
        expert_B_list = [
            [
                [math.cos(0.19 * (e + 1) * (o + 1) - 0.04 * r) for r in range(expert_rank)]
                for o in range(out_features)
            ]
            for e in range(num_experts)
        ]
        router_w = [
            [math.sin(0.23 * (e + 1) * (c + 1)) * 0.25 for c in range(in_features)]
            for e in range(num_experts)
        ]
        router = ProgressiveModalityRouterPq(
            router_weights=router_w,
            router_bias=[0.1, -0.1, 0.2, 0.0],
            warmup_ratio=0.25,
            transition_ratio=0.50,
        )
        layer = MoELoRAwithSharedPq(
            shared_A=shared_A,
            shared_B=shared_B,
            expert_A_list=expert_A_list,
            expert_B_list=expert_B_list,
            router=router,
        )

        x_seq = [
            [math.sin(0.05 * (t + 1) * (c + 1)) for c in range(in_features)]
            for t in range(3)
        ]
        # Audio + Video present: mask = [0, 1, 1, 0]
        modality_mask = [0, 1, 1, 0]

        # Warmup phase (progress=0.10): strictly follows binary modality_mask
        _, weights_warmup = layer.forward_delta(x_seq, modality_mask, progress=0.10)
        self.assertEqual(weights_warmup, [0, Q16_ONE, Q16_ONE, 0])

        # Synergy phase (progress=1.0): cross-modal collaboration active on all experts
        delta_synergy, weights_synergy = layer.forward_delta(
            x_seq, modality_mask, progress=1.0
        )
        self.assertEqual(len(delta_synergy), 3)
        self.assertEqual(len(delta_synergy[0]), out_features)
        self.assertTrue(all(0 < w < Q16_ONE for w in weights_synergy))


if __name__ == "__main__":
    unittest.main()
