"""Real CPU PEFT/AdamW resume equivalence, with a tiny synthetic decoder.

Tests the same adapter/Head, optimizer, scheduler, RNG, checkpoint code used
by H200, substituting only Jeff's 31B backbone and official prompt modules.
Run with the isolated CPU Torch/PEFT stack; no patient files or GPU weights.

uv run --with 'torch==2.9.1+cpu' --with 'peft==0.21.1' \
  --with 'transformers==5.17.0' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m scripts.check_jeff_resume_cpu
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from tcmdiscern.jeff_gpu_core import decision_logits
    from tcmdiscern.jeff_losses import torch_loss
    from tcmdiscern.jeff_resume import restore_resume, save_resume

    class SmallBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(40, 16)
            self.q_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True):
            return types.SimpleNamespace(
                last_hidden_state=self.q_proj(self.embed_tokens(input_ids))
            )

    class SmallHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("base_rows", torch.randn(16, 16) * 0.02)
            self.register_buffer("mu", torch.zeros(16))
            self.register_buffer("sigma", torch.ones(16))
            self.delta = torch.nn.Linear(16, 16)

        def forward(self, h):
            return 30 * torch.tanh(h @ self.base_rows.T / 30) + self.delta(
                (h - self.mu) / self.sigma
            )

    def load_small_head(folder):
        head = SmallHead()
        head.load_state_dict(load_file(str(Path(folder) / "head.safetensors")), strict=True)
        return head

    # Simulate the upstream file readers, keeping the actual checkpoint,
    # Torch optimizer, LoRA, and autograd implementations unchanged.
    mocked_root = types.ModuleType("jeff")
    mocked_head = types.ModuleType("jeff.head")
    mocked_head.load_head = load_small_head
    mocked_prompt = types.ModuleType("jeff.prompting")
    mocked_prompt.encode_prompt = lambda _tok, _row, _limit: [2, 5, 7, 9]
    sys.modules["jeff"] = mocked_root
    sys.modules["jeff.head"] = mocked_head
    sys.modules["jeff.prompting"] = mocked_prompt

    torch.manual_seed(220261112)
    adapter = get_peft_model(
        SmallBackbone(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0.0),
    )
    head = SmallHead()
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT adapter did not expose trainable parameters")
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": 0.001}, {"params": list(head.delta.parameters()), "lr": 0.001}]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    ids = [2, 5, 7, 9]

    def one_update():
        adapter.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        scores = decision_logits(adapter, head, ids, device=torch.device("cpu"))
        loss = torch_loss(scores, positive_indices=[1, 3], gold_index=1, method="s3")
        loss.backward()
        optimizer.step()
        scheduler.step()

    def model_logits():
        adapter.eval()
        head.eval()
        with torch.inference_mode():
            return (
                decision_logits(adapter, head, ids, device=torch.device("cpu"))
                .detach()
                .float()
                .clone()
            )

    with TemporaryDirectory(prefix="tcmdiscern_resume_") as folder:
        folder = Path(folder)
        original_model_dir = folder / "released"
        original_model_dir.mkdir()
        (original_model_dir / "head_config.json").write_text(
            '{"kind":"residual","k_slots":16}', encoding="utf-8"
        )
        run_dir = folder / "run"
        one_update()
        first = model_logits()
        manifest = {
            "run_id": "synthetic",
            "global_update": 1,
            "cursor_epoch": 0,
            "cursor_offset": 4,
            "seed": 17,
            "method": "s3",
        }
        checkpoint = save_resume(
            run_dir,
            state=manifest,
            model_dir=original_model_dir,
            adapter=adapter,
            head=head,
            backbone=adapter,
            tokenizer=None,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        expected_random = torch.rand(3)
        one_update()
        second = model_logits()
        torch.rand(50)

        recovered = restore_resume(
            checkpoint,
            expected={"run_id": "synthetic", "seed": 17, "method": "s3"},
            adapter=adapter,
            head=head,
            backbone=adapter,
            tokenizer=None,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        restored_first = model_logits()
        if not torch.allclose(first, restored_first, atol=1e-5, rtol=0):
            raise RuntimeError("Restored first-step logits differ")
        if not torch.allclose(expected_random, torch.rand(3), atol=0, rtol=0):
            raise RuntimeError("RNG state was not restored")
        one_update()
        restored_second = model_logits()
        error = float((second - restored_second).abs().max())
        if error > 1e-5:
            raise RuntimeError(
                f"Resume then second optimizer update differs from uninterrupted training: {error}"
            )
        if recovered["global_update"] != 1:
            raise RuntimeError("Resume cursor step changed")

    report = {
        "date": "2026-11-12",
        "status": "PEFT_CPU_RESTART_MATCHES_UNINTERRUPTED_TRAINING",
        "model": "synthetic tiny decoder/Head, no Jeff weights",
        "first_optimizer_update_saved": True,
        "resumed_optimizer_state": True,
        "resumed_scheduler_state": True,
        "resumed_torch_rng": True,
        "replayed_second_update_matches": True,
        "step2_max_logit_difference": error,
        "patient_data_loaded": False,
        "actual_h200_test": False,
    }
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/jeff_cpu_resume_2026-11-12.json"
    with out.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
