"""Session and registered-matrix isolation without allocating 31B or CUDA."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tcmdiscern.h200.session import JeffGPUSession
from tcmdiscern.h200.workflow import H200Runner


def test_peft_unload_clears_stale_config_and_preserves_original_backbone(
    monkeypatch,
    tmp_path,
):
    """PEFT BaseTuner retains peft_config after unloading. Catch regression."""
    from tcmdiscern.jeff_checkpoint import reset_released_backbone

    class FakeTunerLayer:
        pass

    class OriginalBackbone:
        def __init__(self):
            self.peft_config = {"default": True}

        def modules(self):
            return (self,)

        def eval(self):
            return self

    class FakePeftModel:
        def __init__(self, original):
            self.base_model = SimpleNamespace(unload=lambda: original)

    original = OriginalBackbone()
    adapter = FakePeftModel(original)
    loaded = []
    jeff = ModuleType("jeff")
    jeff_head = ModuleType("jeff.head")

    class Head:
        def to(self, device):
            return self

        def eval(self):
            return self

    jeff_head.load_head = lambda folder: (loaded.append(folder), Head())[1]
    jeff.head = jeff_head
    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel
    tuners = ModuleType("peft.tuners")
    utils = ModuleType("peft.tuners.tuners_utils")
    utils.BaseTunerLayer = FakeTunerLayer
    monkeypatch.setitem(sys.modules, "jeff", jeff)
    monkeypatch.setitem(sys.modules, "jeff.head", jeff_head)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners", tuners)
    monkeypatch.setitem(sys.modules, "peft.tuners.tuners_utils", utils)

    decider = SimpleNamespace(
        base=adapter,
        forward=adapter,
        model=SimpleNamespace(model=adapter, eval=lambda: None),
        device="cuda",
        head=None,
    )
    status = reset_released_backbone(decider, tmp_path)
    assert status["released_backbone_reused"] is True
    assert decider.base is decider.forward is decider.model.model is original
    assert not hasattr(original, "peft_config")
    assert loaded == [tmp_path]


def test_peft_reset_rejects_any_remaining_tuner_layers(monkeypatch, tmp_path):
    from tcmdiscern.jeff_checkpoint import reset_released_backbone

    class FakeTunerLayer:
        pass

    class Original:
        def __init__(self):
            self.peft_config = {"default": True}

        def modules(self):
            return (self, FakeTunerLayer())

    class FakePeftModel:
        def __init__(self, original):
            self.base_model = SimpleNamespace(unload=lambda: original)

    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel
    utils = ModuleType("peft.tuners.tuners_utils")
    utils.BaseTunerLayer = FakeTunerLayer
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners", ModuleType("peft.tuners"))
    monkeypatch.setitem(sys.modules, "peft.tuners.tuners_utils", utils)
    jeff = ModuleType("jeff")
    jeff_head = ModuleType("jeff.head")
    jeff_head.load_head = lambda _: None
    monkeypatch.setitem(sys.modules, "jeff", jeff)
    monkeypatch.setitem(sys.modules, "jeff.head", jeff_head)

    original = Original()
    decider = SimpleNamespace(base=FakePeftModel(original))
    with pytest.raises(RuntimeError, match="left real tuner layers"):
        reset_released_backbone(decider, tmp_path)
    assert hasattr(original, "peft_config")


def test_session_lazy_loads_once_for_multiple_original_calls(monkeypatch, tmp_path):
    import scripts.train_jeff_lora as original_loader

    count = [0]
    backbone = object()
    decider = SimpleNamespace(base=backbone, model=SimpleNamespace(model=backbone))
    model = SimpleNamespace(decider=decider)

    def load(_):
        count[0] += 1
        return model

    monkeypatch.setattr(original_loader, "load_original_checkpoint", load)
    session = JeffGPUSession(tmp_path)
    assert session.released() is model
    assert session.original() is model
    assert session.original() is model
    assert session.model_weight_loads == count[0] == 1


def test_session_unloads_adapter_before_next_training(monkeypatch, tmp_path):
    import scripts.train_jeff_lora as original_loader
    import tcmdiscern.h200.session as session_code
    import tcmdiscern.jeff_train_loop as loop

    backbone = object()
    adapter = SimpleNamespace(peft_config={"default": True})
    decider = SimpleNamespace(base=backbone, model=SimpleNamespace(model=backbone))
    model = SimpleNamespace(decider=decider)
    loads = []
    resets = []
    modes = []

    monkeypatch.setattr(
        original_loader,
        "load_original_checkpoint",
        lambda _: (loads.append(True), model)[1],
    )

    def reset(decider, original_dir):
        resets.append(True)
        decider.base = backbone
        decider.model.model = backbone

    def train(args, *, released):
        assert released is model
        assert decider.base is backbone
        decider.base = adapter
        decider.model.model = adapter
        modes.append(args.study_mode)
        return {
            "global_update": 1,
            "loss_method": "s1",
            "peak_cuda_bytes": 10,
            "run_id": args.run_id,
        }

    monkeypatch.setattr(session_code, "reset_released_backbone", reset)
    monkeypatch.setattr(loop, "run", train)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    session = JeffGPUSession(tmp_path)
    for mode in ("disease16", "pair16"):
        session.train_cli(
            (
                "-m",
                "scripts.run_jeff_finetune",
                "--model-dir",
                str(tmp_path),
                "--method",
                "s1",
                "--study-mode",
                mode,
                "--run-id",
                mode,
            )
        )
    assert modes == ["disease16", "pair16"]
    assert len(resets) == 1
    assert len(loads) == session.model_weight_loads == 1


def test_registered_screen_uses_single_session_for_all_gpu_stages(monkeypatch, tmp_path):
    runner = H200Runner(root=tmp_path, dry_run=True)
    seen = []
    monkeypatch.setattr(runner, "stage", lambda stage: seen.append(stage))
    session = SimpleNamespace()
    runner._primary_stages(session=session, full=False)
    assert len(seen) == 10  # one released baseline, three trainings, six Dev checkpoints
    assert len({stage.name for stage in seen}) == 10
    assert all(stage.perform is not None for stage in seen)
    assert all(stage.verify is not None for stage in seen)


def test_registered_full_reuses_first_seed_and_preserves_three_seed_budget(monkeypatch, tmp_path):
    runner = H200Runner(root=tmp_path, dry_run=True)
    seen = []
    monkeypatch.setattr(runner, "stage", lambda stage: seen.append(stage))
    runner._primary_stages(session=SimpleNamespace(), full=True)
    assert len(seen) == 28  # one baseline, 9 trainings, 18 Dev checkpoints
    assert len({stage.name for stage in seen}) == 28
    assert all(stage.perform is not None and stage.verify is not None for stage in seen)
