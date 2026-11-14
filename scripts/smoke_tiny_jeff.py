"""Run the released Jeff software path with a tiny random Gemma 4 instead of 31B.

This smoke is not an accuracy experiment: it proves attention, PEFT, Loss,
optimizer changes and serialization on an actual PyTorch model in minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from tcmdiscern.jeff_config import JEFF_BIDIRECTIONAL, JEFF_MAX_INPUT_TOKENS
from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_release_attention import inspect_release_attention

ROOT = Path(__file__).resolve().parents[1]
MINI_VOCAB_SIZE = 512
MINI_HIDDEN_SIZE = 64
MINI_LAYER_TYPES = (
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
)


def read_training_probe(root: Path) -> dict:
    """Read a real ambiguous Train example, never a Dev or Test label."""
    path = root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Prepare Train artifacts first: {path}")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if len(item.get("positive_indices", [])) >= 2:
                if len(item["row"]["options"]) != 16:
                    raise RuntimeError("Train example does not have 16 candidates")
                if item["target_index"] not in item["positive_indices"]:
                    raise RuntimeError("Train gold index is not in positive indices")
                return reshuffle_training_example(item, epoch=0, seed=20261112)
    raise RuntimeError("Cannot find an original multi-positive Train example")


def miniature_text_config(metadata: dict):
    """Preserve Gemma 4 attention varieties, PLE and shared K/V in 4 layers."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    if metadata["release_jeff_bidirectional"] != JEFF_BIDIRECTIONAL:
        raise RuntimeError("Unexpected Jeff release bidirectional setting")
    return Gemma4TextConfig(
        vocab_size=MINI_VOCAB_SIZE,
        hidden_size=MINI_HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=len(MINI_LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        layer_types=list(MINI_LAYER_TYPES),
        num_kv_shared_layers=1,
        attention_k_eq_v=metadata["attention_k_eq_v"],
        hidden_size_per_layer_input=8 if metadata["has_per_layer_embeddings"] else 0,
        vocab_size_per_layer_input=MINI_VOCAB_SIZE,
        use_bidirectional_attention=metadata["gemma4_native_attention"],
        max_position_embeddings=256,
        sliding_window=32,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def miniature_token_ids(real_ids: list[int], width: int = 64) -> list[int]:
    """Keep SemIf token order while mapping IDs into the random tiny vocabulary."""
    if len(real_ids) < 40 or width < 40:
        raise ValueError("The official prompt or miniature window is too short")
    return [1 + int(token) % (MINI_VOCAB_SIZE - 1) for token in real_ids[-width:]]


def verify_mini_train_and_reload(decider, ids: list[int], example: dict, directory: Path) -> dict:
    """Train a real PEFT adapter and official Jeff Head on tiny weights."""
    import torch
    from jeff.head import DecisionHead
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file, save_file

    from scripts.train_jeff_lora import inject_lora_and_assert_step_zero
    from tcmdiscern.jeff_gpu_core import (
        capture_first_update,
        gradient_audit,
        parameter_audit,
        supervised_loss,
        trainable_parameters,
        verify_first_update,
    )

    adapter, _, step0_max, counts, trainable = inject_lora_and_assert_step_zero(
        decider,
        ids,
        rank=4,
        alpha=8,
    )
    print(f"[MINI] LoRA Step-0 PASS; modules={counts}; parameters={trainable}", flush=True)
    parameter_audit(adapter, decider.head)
    lora_parameters, head_parameters = trainable_parameters(adapter, decider.head)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": 1e-3},
            {"params": head_parameters, "lr": 1e-3},
        ],
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    first_update = None
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        decider.model.train()
        decider.head.train()
        output = decider.forward(
            input_ids=torch.tensor([ids], device=decider.device),
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, -1]
        logits = decider.head(hidden.to(torch.bfloat16).float())[0, :16]
        loss = supervised_loss(logits, example, "s3")
        if not torch.isfinite(loss).item():
            raise RuntimeError("Mini Gemma 4 Set Loss is non-finite")
        loss.backward()
        gradient_audit(adapter, decider.head)
        torch.nn.utils.clip_grad_norm_(
            [*lora_parameters, *head_parameters],
            1.0,
            error_if_nonfinite=True,
        )
        before_step = capture_first_update(adapter, decider.head) if step == 0 else None
        optimizer.step()
        scheduler.step()
        if before_step is not None:
            first_update = verify_first_update(before_step)
        losses.append(float(loss.detach().item()))
        print(
            f"[MINI] Optimizer update {step + 1}/2 PASS; loss={losses[-1]:.5f}",
            flush=True,
        )

    decider.model.eval()
    decider.head.eval()
    with torch.inference_mode():
        expected = (
            decider.head(decider.hidden(ids).to(torch.bfloat16).float())[0, :16]
            .detach()
            .float()
            .cpu()
        )
    directory.mkdir(parents=True, exist_ok=False)
    adapter.save_pretrained(directory / "adapter", safe_serialization=True)
    save_file(
        {
            name: item.detach().cpu().contiguous()
            for name, item in decider.head.state_dict().items()
        },
        str(directory / "head.safetensors"),
    )
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save(scheduler.state_dict(), directory / "scheduler.pt")
    state = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    if len(state["param_groups"]) != 2 or not state["state"]:
        raise RuntimeError("Mini optimizer snapshot has no trainable state")

    original = adapter.base_model.unload()
    if any(isinstance(module, BaseTunerLayer) for module in original.modules()):
        raise RuntimeError("Mini PEFT unload left active LoRA modules")
    if hasattr(original, "peft_config"):
        delattr(original, "peft_config")
    decider.model.model = original
    decider.base = original
    decider.forward = original
    restored = PeftModel.from_pretrained(original, directory / "adapter", is_trainable=False)
    decider.model.model = restored
    decider.base = restored
    decider.forward = restored
    head = DecisionHead(MINI_HIDDEN_SIZE, 16, softcap=30.0).to(decider.device)
    head.load_state_dict(load_file(str(directory / "head.safetensors")), strict=True)
    decider.head = head.eval()
    decider.model.eval()
    with torch.inference_mode():
        actual = (
            decider.head(decider.hidden(ids).to(torch.bfloat16).float())[0, :16]
            .detach()
            .float()
            .cpu()
        )
    if not torch.allclose(actual, expected, atol=1e-3, rtol=1e-4):
        raise RuntimeError(
            "Mini PEFT+Head checkpoint reload changed raw logits: "
            f"max_abs={(actual - expected).abs().max().item():.6g}"
        )
    print("[MINI] PEFT, Head, Optimizer save + fresh reload PASS", flush=True)
    return {
        "optimizer_updates": 2,
        "loss_values": losses,
        "step0_max_abs_difference": step0_max,
        "lora_trainable_parameters": trainable,
        "first_optimizer_update_verified": first_update,
        "fresh_checkpoint_logits_match": True,
        "optimizer_state_saved": True,
    }


def run(*, model_dir: Path, root: Path = ROOT) -> dict:
    """Run a tiny Gemma4 model; never map the released 31B weights."""
    model_dir = model_dir.expanduser().resolve(strict=True)
    metadata = inspect_release_attention(model_dir)
    print("[MINI] Released config: " + json.dumps(metadata, ensure_ascii=False), flush=True)
    import torch
    from jeff.head import DecisionHead
    from jeff.model import Decider
    from jeff.prompting import encode_prompt
    from transformers import AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from tcmdiscern.jeff_attention import (
        audit_bidirectional_determinism,
        audit_with_safe_backend,
    )
    from tcmdiscern.jeff_release_attention import runtime_attention_contract

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    torch.manual_seed(42)
    question = read_training_probe(root)
    encoder = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    original_ids = encode_prompt(encoder, question["row"], JEFF_MAX_INPUT_TOKENS)
    ids = miniature_token_ids(original_ids)
    model = (
        Gemma4ForCausalLM(miniature_text_config(metadata))
        .to(
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        .eval()
    )
    head = DecisionHead(MINI_HIDDEN_SIZE, 16, softcap=30.0).float()
    decider = Decider(
        model,
        encoder,
        head,
        max_tokens=256,
        bidirectional=metadata["release_jeff_bidirectional"],
        long_input="refuse",
    )
    contract = runtime_attention_contract(decider)
    if contract != metadata["expected_attention"]:
        raise RuntimeError("Miniature attention does not match released model metadata")
    if contract == "causal":
        attention = audit_with_safe_backend(decider, ids)
    else:
        attention = audit_bidirectional_determinism(decider, ids)
        attention["attention_backend"] = "released_default"
    decider.attention_backend = attention["attention_backend"]
    print(f"[MINI] {contract} attention PASS", flush=True)
    output_dir = root / "data/h200"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "micro_smoke.json"
    report_file.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mini-jeff-", dir=output_dir) as folder:
        trained = verify_mini_train_and_reload(
            decider,
            ids,
            question,
            Path(folder) / "checkpoint",
        )
    report = {
        "status": "MINI_JEFF_PIPELINE_PASS",
        "model_weight_loads": 0,
        "model_type": "four_layer_random_gemma4",
        "non_benchmark": True,
        "test_opened": False,
        "attention_contract": contract,
        "attention_backend": attention["attention_backend"],
        "real_train_prompt_tokens": len(original_ids),
        "miniature_prompt_tokens": len(ids),
        "train_observed_positives": len(question["positive_indices"]),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        **trained,
    }
    fd, temporary = tempfile.mkstemp(prefix=".mini-report-", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, report_file)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("[MINI] " + json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models/Jeff-1.1-Large")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        print(json.dumps(inspect_release_attention(args.model_dir), ensure_ascii=False, indent=2))
        return
    run(model_dir=args.model_dir)


if __name__ == "__main__":
    main()
