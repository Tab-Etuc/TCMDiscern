"""CPU-only miniature validation of the real Jeff->PEFT->Head gradient path.

This uses synthetic tiny weights. It tests PEFT injection and LoRA gradient
propagation without the original 31B checkpoint or any patient information.

Run with a temporary CPU PyTorch/PEFT installation; do not change base deps.
"""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tcmdiscern.jeff_gpu_core import (
    decision_logits,
    gradient_audit,
    parameter_audit,
    supervised_loss,
    validate_step0,
)


def main() -> None:
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file

    from scripts.train_jeff_lora import (
        inject_lora_and_assert_step_zero,
        original_logits,
    )

    class SmallDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(100, 32)
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                setattr(self, name, torch.nn.Linear(32, 32))
            self.norm = torch.nn.LayerNorm(32)
            self.config = {"model_type": "tiny-test"}

        def forward(self, *, input_ids, attention_mask=None, use_cache=False, return_dict=True):
            x = self.emb(input_ids)
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                x = x + torch.tanh(getattr(self, name)(x)) * 0.1
            return SimpleNamespace(last_hidden_state=self.norm(x))

    class SmallCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SmallDecoder()

    class SmallHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("base_rows", torch.randn(16, 32) * 0.02)
            self.register_buffer("mu", torch.zeros(32))
            self.register_buffer("sigma", torch.ones(32))
            self.delta = torch.nn.Linear(32, 16)

        def forward(self, h):
            return 30.0 * torch.tanh(h @ self.base_rows.T / 30.0) + self.delta(
                (h - self.mu) / self.sigma
            )

    class Decider:
        def __init__(self):
            self.model = SmallCausalLM()
            self.base = self.model.model
            self.forward = self.base
            self.head = SmallHead()
            self.device = torch.device("cpu")

        def hidden(self, ids):
            return self.forward(
                input_ids=torch.tensor([ids]),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1]

    results = {}
    torch.manual_seed(2)
    for method in ("s1", "s2", "s3", "s4"):
        decider = Decider()
        released_decoder = deepcopy(decider.base)
        ids = [1, 3, 8, 15, 42]
        adapter, initial, difference, modules, n = inject_lora_and_assert_step_zero(
            decider, ids, rank=4, alpha=8
        )
        checks = validate_step0(initial.tolist(), original_logits(decider, ids))
        report = parameter_audit(adapter, decider.head)
        if any(value != 1 for value in modules.values()):
            raise ValueError("Tiny model did not inject into all seven projections")
        if n <= 0:
            raise ValueError("No new LoRA parameters")
        adapter.train()
        decider.head.train()
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": 0.01},
                {"params": list(decider.head.delta.parameters()), "lr": 0.01},
            ]
        )
        positive = [1, 5, 9] if method != "s1" else [5]
        item = {"target_index": 5, "positive_indices": positive}
        logits = decision_logits(decider.forward, decider.head, ids, device=decider.device)
        loss = supervised_loss(logits, item, method)
        optimizer.zero_grad()
        loss.backward()
        gradients = gradient_audit(adapter, decider.head)
        optimizer.step()
        with torch.no_grad():
            updated = original_logits(decider, ids)
        if torch.allclose(initial, updated, rtol=0, atol=1e-5):
            raise RuntimeError("Toy fine-tuning did not change head predictions")

        # Verify adapter and released-Head state round-trip after corruption.
        adapter.eval()
        decider.head.eval()
        expected = original_logits(decider, ids).detach()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            adapter.save_pretrained(directory / "adapter", safe_serialization=True)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in decider.head.state_dict().items()},
                str(directory / "head.safetensors"),
            )
            with torch.no_grad():
                decider.head.delta.bias.add_(1.0)
                next(
                    p
                    for key, p in adapter.named_parameters()
                    if p.requires_grad and "lora_B" in key
                ).add_(0.1)
            state = load_file(str(directory / "adapter/adapter_model.safetensors"))
            set_peft_model_state_dict(adapter, state, adapter_name="default")
            decider.head.load_state_dict(
                load_file(str(directory / "head.safetensors")), strict=True
            )
            restored = original_logits(decider, ids)
            if not torch.allclose(expected, restored, rtol=1e-5, atol=1e-5):
                raise RuntimeError("Toy model adapter+Head checkpoint did not reload")
            fresh_adapter = PeftModel.from_pretrained(
                released_decoder, directory / "adapter", is_trainable=False
            )
            fresh_head = SmallHead().eval()
            fresh_head.load_state_dict(load_file(str(directory / "head.safetensors")), strict=True)
            fresh_output = decision_logits(
                fresh_adapter, fresh_head, ids, device=decider.device
            ).detach()
            if not torch.allclose(expected, fresh_output, rtol=1e-5, atol=1e-5):
                raise RuntimeError("Fresh PEFT adapter+Head restore differs from saved logits")
        results[method] = {
            "initial_logit_diff": difference,
            "step0_equal": checks["all_16_logits_match"],
            "adapter_gradients": gradients["adapter_gradients_present"],
            "head_gradients": gradients["head_delta_gradients_present"],
            "adapter_parameters": report["trainable_adapter_parameters"],
            "checkpoint_roundtrip": True,
        }
    print(json.dumps({"status": "TINY_PEFT_CPU_FOUR_METHODS_PASS", "methods": results}, indent=2))


if __name__ == "__main__":
    main()
