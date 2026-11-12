"""Run a tiny synthetic CPU model through the exact Jeff LoRA injection path.

This exercises the new PEFT adapter, 16-slot head, 4 supervised losses,
autograd, optimizer update and adapter reload without loading 31B weights.
It is a preflight for code correctness, not a substitute for H200 parity.

uv run --with 'torch==2.9.1+cpu' --with 'peft==0.21.1' \
  --with 'transformers==5.17.0' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m scripts.check_jeff_peft_toy
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.train_jeff_lora import inject_lora_and_assert_step_zero
from tcmdiscern.jeff_losses import torch_loss


def main() -> None:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    torch.manual_seed(20261112)

    class TinyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(64, 16)
            self.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.k_proj = torch.nn.Linear(16, 16, bias=False)
            self.v_proj = torch.nn.Linear(16, 16, bias=False)
            self.o_proj = torch.nn.Linear(16, 16, bias=False)
            self.gate_proj = torch.nn.Linear(16, 16, bias=False)
            self.up_proj = torch.nn.Linear(16, 16, bias=False)
            self.down_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True):
            hidden = self.embed_tokens(input_ids)
            hidden = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
            hidden = self.o_proj(hidden) + self.down_proj(
                torch.relu(self.gate_proj(hidden)) * self.up_proj(hidden)
            )
            return SimpleNamespace(last_hidden_state=hidden)

    class TinyFullModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyBackbone()

    class TinyHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta = torch.nn.Linear(16, 16)

        def forward(self, h):
            return self.delta(h)

    class TinyDecider:
        def __init__(self) -> None:
            self.model = TinyFullModel()
            self.base = self.model.model
            self.forward = self.base
            self.head = TinyHead()
            self.device = torch.device("cpu")

        def hidden(self, ids):
            return self.forward(
                input_ids=torch.tensor([ids]),
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1]

    decider = TinyDecider()
    ids = [1, 5, 3, 16, 9]
    adapter, _, max_parity, target_counts, trainable = inject_lora_and_assert_step_zero(
        decider, ids, rank=2, alpha=4
    )
    if max_parity > 1e-3 or trainable == 0:
        raise RuntimeError("PEFT initial adapter changes original model output")
    assert target_counts == {name: 1 for name in target_counts}
    decider.model.train()
    decider.head.train()
    head_parameters = list(decider.head.parameters())
    lora_parameters = [param for param in adapter.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW([*lora_parameters, *head_parameters], lr=0.001)
    results = {}
    for method in ("s1", "s2", "s3", "s4"):
        optimizer.zero_grad(set_to_none=True)
        h = decider.hidden(ids)
        logits = decider.head(h.to(torch.bfloat16).float())[0]
        positives = [1] if method == "s1" else [1, 5, 9]
        loss = torch_loss(logits, positive_indices=positives, gold_index=1, method=method)
        loss.backward()
        lora_grad = sum(
            float(param.grad.abs().sum()) for param in lora_parameters if param.grad is not None
        )
        head_grad = sum(
            float(param.grad.abs().sum()) for param in head_parameters if param.grad is not None
        )
        if lora_grad <= 0 or head_grad <= 0:
            raise RuntimeError(f"Toy {method} did not backpropagate through both trainable parts")
        optimizer.step()
        results[method] = {
            "loss": float(loss.detach()),
            "lora_gradient_l1": lora_grad,
            "head_gradient_l1": head_grad,
            "gradient_path_verified": True,
        }

    decider.model.eval()
    decider.head.eval()
    with torch.no_grad():
        saved_head = {
            key: value.detach().clone() for key, value in decider.head.state_dict().items()
        }
        reference = decider.head(decider.hidden(ids).to(torch.bfloat16).float())
    with TemporaryDirectory(prefix="jeff_peft_cpu_") as directory:
        adapter.save_pretrained(directory, safe_serialization=True)
        weights = Path(directory) / "adapter_model.safetensors"
        if not weights.exists():
            raise RuntimeError("PEFT did not save adapter_model.safetensors")
        with torch.no_grad():
            next(
                p for name, p in adapter.named_parameters() if p.requires_grad and "lora_B" in name
            ).add_(0.1)
            decider.head.delta.bias.add_(0.3)
        with torch.no_grad():
            corrupted = decider.head(decider.hidden(ids).to(torch.bfloat16).float())
        if torch.allclose(reference, corrupted, atol=1e-4):
            raise RuntimeError("Toy corruption probe failed")
        result = set_peft_model_state_dict(adapter, load_file(str(weights)), adapter_name="default")
        if getattr(result, "unexpected_keys", []):
            raise RuntimeError(f"Unexpected adapter keys when loading: {result}")
        decider.head.load_state_dict(saved_head, strict=True)
        with torch.no_grad():
            restored = decider.head(decider.hidden(ids).to(torch.bfloat16).float())
        if not torch.allclose(reference, restored, atol=1e-3, rtol=0):
            raise RuntimeError("Toy LoRA adapter save/reload changed the model outputs")

    report = {
        "date": "2026-11-12",
        "status": "TINY_CPU_PEFT_AND_HEAD_GRADIENT_CHECKPOINT_PASS",
        "torch": torch.__version__,
        "training_real_jeff": False,
        "original_jeff_head_loaded": False,
        "target_modules": target_counts,
        "lora_trainable_parameters": trainable,
        "step0_max_abs_logit_difference": max_parity,
        "all_four_losses_have_nonzero_lora_and_head_gradient": True,
        "adapter_checkpoint_reload_matches": True,
        "losses": results,
    }
    root = Path(__file__).resolve().parents[1]
    target = root / "experiments/jeff_toy_peft_all_projections_2026-11-12.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
