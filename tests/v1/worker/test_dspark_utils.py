# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _FakeHFConfig:
    model_type: str


@dataclass
class _FakeModelConfig:
    hf_config: _FakeHFConfig


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


@pytest.mark.parametrize(
    ("model_type", "enable_eplb", "num_redundant_experts", "shares_eplb_config"),
    [
        ("deepseek_v4", True, 32, True),
        ("deepseek_v41", False, 0, False),
    ],
)
def test_dspark_parallel_config_eplb_by_model_type(
    model_type: str,
    enable_eplb: bool,
    num_redundant_experts: int,
    shares_eplb_config: bool,
):
    target_config = _FakeParallelConfig()
    draft_model_config = _FakeModelConfig(
        hf_config=_FakeHFConfig(model_type=model_type)
    )

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
        draft_model_config=draft_model_config,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert draft_config.enable_eplb is enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == num_redundant_experts
    assert not draft_config.enable_elastic_ep
    if shares_eplb_config:
        assert draft_config.eplb_config is target_config.eplb_config
    else:
        assert draft_config.eplb_config is not target_config.eplb_config


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.speculative_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)


@pytest.mark.parametrize(
    "model_type,mixed,expected",
    [("qwen3_5_moe", False, True), ("qwen3_5", True, False), ("other", False, False)],
)
def test_qwen_dspark_tags_only_complete_draft_windows(
    monkeypatch, model_type, mixed, expected
):
    shape = dict(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    window = SlidingWindowSpec(sliding_window=4096, **shape)
    other = FullAttentionSpec(**shape) if mixed else window
    monkeypatch.setattr(
        model_runner,
        "get_kv_cache_spec",
        lambda _: {"window": window, "draft.0": window, "draft.1": other},
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.model_config = _FakeModelConfig(_FakeHFConfig(model_type))
    runner.speculative_config = SimpleNamespace(use_dspark=lambda: True)
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.draft_attn_layer_names = {"draft.0", "draft.1"}
    runner.speculator.dcp_size = runner.dcp_size = 1

    specs = runner.get_kv_cache_spec()
    assert not specs["window"].is_dspark_window
    assert specs["draft.0"].is_dspark_window == expected
    assert getattr(specs["draft.1"], "is_dspark_window", False) == expected


@pytest.mark.parametrize(
    "excluded", [None, "connector", "target_window", "other_draft"]
)
def test_verified_prompt_requires_local_context_only_draft(monkeypatch, excluded):
    shape = dict(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    window = SlidingWindowSpec(sliding_window=4096, **shape)
    layers = {
        "target": window if excluded == "target_window" else FullAttentionSpec(**shape),
        "state": MambaSpec(
            block_size=16,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        "draft": window,
    }
    monkeypatch.setattr(model_runner, "get_kv_cache_spec", lambda _: dict(layers))
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=object() if excluded == "connector" else None,
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, kv_offloading_size=None
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    runner.model_config = _FakeModelConfig(_FakeHFConfig("qwen3_5_moe"))
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1, prefill_context_parallel_size=1
    )
    arch = "OtherDSpark" if excluded == "other_draft" else "Qwen3DSparkModel"
    runner.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        use_eagle_block_drop=lambda: True,
        draft_model_config=SimpleNamespace(architectures=[arch]),
    )
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.draft_attn_layer_names = {"draft"}
    runner.speculator.dcp_size = runner.dcp_size = 1

    assert runner.get_kv_cache_spec()["state"].dspark_verified_prompt == (
        excluded is None
    )
