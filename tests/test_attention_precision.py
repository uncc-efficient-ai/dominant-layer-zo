"""Regression coverage for fp16 QK overflow and the public loader option."""
import copy
import tempfile
import unittest

import torch
from transformers import LlamaConfig

from dominant_layer_zo.config import DEFAULTS, validate
from dominant_layer_zo.models import backend, load_model
from test_core import tiny, tokenizer


class AttentionPrecisionTests(unittest.TestCase):
    def test_fp16_score_overflow_is_fixed_without_promoting_weights(self):
        cfg = LlamaConfig(hidden_size=16, intermediate_size=24,
                          num_attention_heads=2, num_key_value_heads=2,
                          attention_dropout=0., max_position_embeddings=16)
        attention = backend('llama2', 'modeling').LlamaAttention(cfg, layer_idx=0).half().eval()
        with torch.no_grad():
            eye = torch.eye(16, dtype=torch.float16)
            attention.q_proj.weight.copy_(eye * 100)
            attention.k_proj.weight.copy_(eye * 100)
            attention.v_proj.weight.copy_(eye)
            attention.o_proj.weight.copy_(eye)
            hidden = torch.full((1, 2, 16), 2., dtype=torch.float16)
            positions = torch.zeros((1, 2), dtype=torch.long)
            legacy, _, _ = attention(hidden, position_ids=positions)
            self.assertFalse(torch.isfinite(legacy).all())
            cfg.fp32_attention_scores = True
            fixed, probs, _ = attention(hidden, position_ids=positions, output_attentions=True)
            self.assertTrue(torch.isfinite(fixed).all())
            torch.testing.assert_close(fixed, hidden, rtol=0, atol=0)
            torch.testing.assert_close(probs, torch.full_like(probs, .5), rtol=0, atol=0)
            self.assertEqual(fixed.dtype, torch.float16)
            self.assertTrue(all(p.dtype == torch.float16 for p in attention.parameters()))
            mask = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
            mask[..., 0, 1] = torch.finfo(torch.float16).min
            _, masked_probs, _ = attention(hidden, position_ids=positions,
                                           attention_mask=mask, output_attentions=True)
            self.assertEqual(masked_probs[..., 0, 1].count_nonzero(), 0)

    def test_qwen_promoted_scores_preserve_causal_exclusions(self):
        from types import SimpleNamespace
        from dominant_layer_zo.backends.qwen3.modeling import eager_attention_forward
        module = SimpleNamespace(config=SimpleNamespace(fp32_attention_scores=True),
                                 num_key_value_groups=2, training=False)
        q = torch.full((1, 4, 2, 4), 200., dtype=torch.float16)
        k = torch.full((1, 2, 2, 4), 200., dtype=torch.float16)
        k[:, :, 1] = 400.
        v = torch.zeros_like(k)
        v[:, :, 1] = 1.
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
        mask[..., 0, 1] = torch.finfo(torch.float16).min
        output, weights = eager_attention_forward(module, q, k, v, mask, .5)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(weights[..., 0, 1].count_nonzero(), 0)
        self.assertEqual(output[:, 0].count_nonzero(), 0)

    def test_loader_applies_and_serializes_the_explicit_option(self):
        with tempfile.TemporaryDirectory() as td:
            model = tiny('llama2').half().eval()
            model.save_pretrained(td); tokenizer().save_pretrained(td)
            cfg = copy.deepcopy(DEFAULTS['model'])
            cfg['fp32_attention_scores'] = True
            loaded, _ = load_model(cfg, 'cpu', checkpoint=td)
            self.assertTrue(loaded.config.fp32_attention_scores)
            self.assertTrue(all(layer.self_attn.config.fp32_attention_scores for layer in loaded.model.layers))
            self.assertTrue(all(p.dtype == torch.float16 for p in loaded.parameters()))
            self.assertTrue(loaded.config.to_dict()['fp32_attention_scores'])
            cfg['fp32_attention_scores'] = False
            legacy, _ = load_model(cfg, 'cpu', checkpoint=td)
            self.assertFalse(legacy.config.fp32_attention_scores)

    def test_fp32_forward_is_unchanged(self):
        model = tiny('llama2').eval()
        ids = torch.tensor([[1, 3, 4]])
        with torch.no_grad():
            before = model(ids).logits
            model.config.fp32_attention_scores = True
            after = model(ids).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_option_is_boolean(self):
        self.assertFalse(DEFAULTS['model']['fp32_attention_scores'])
        for changes in [{'fp32_attention_scores': 'true'}]:
            with self.subTest(changes=changes):
                cfg = copy.deepcopy(DEFAULTS)
                cfg['model'].update(changes)
                with self.assertRaises(ValueError):
                    validate(cfg)


if __name__ == '__main__':
    unittest.main()
