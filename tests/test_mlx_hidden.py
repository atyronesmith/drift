"""Tests for hidden-state probing (drift.probing.mlx_hidden).

Unit tests cover pure logic (serialization, position/layer resolution) and
run anywhere. Integration tests require Apple Silicon + a small MLX model
and are gated behind DRIFT_MLX_INTEGRATION=1.
"""

import json
import os

import pytest

from drift.probing.mlx_hidden import (
    HiddenStateResult,
    _resolve_layers,
    _resolve_positions,
)

INTEGRATION = os.environ.get("DRIFT_MLX_INTEGRATION") == "1"
SMALL_MODEL = os.environ.get(
    "DRIFT_MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
)


def _make_result() -> HiddenStateResult:
    return HiddenStateResult(
        model="test-model",
        n_input=4,
        layers=[0, 2],
        positions=[3],
        states={0: {3: [0.1] * 8}, 2: {3: [0.25] * 8}},
        input_tokens=["a", "b", "c", "d"],
        content=None,
    )


class TestSerialization:
    def test_roundtrip(self):
        r = _make_result()
        payload = json.dumps(r.to_dict())
        r2 = HiddenStateResult.from_dict(json.loads(payload))
        assert r2.model == r.model
        assert r2.n_input == r.n_input
        assert r2.layers == r.layers
        assert r2.positions == r.positions
        assert r2.input_tokens == r.input_tokens
        assert r2.content is None

    def test_roundtrip_restores_int_keys(self):
        # JSON forces dict keys to strings; from_dict must restore ints
        r = _make_result()
        r2 = HiddenStateResult.from_dict(json.loads(json.dumps(r.to_dict())))
        assert 2 in r2.states
        assert 3 in r2.states[2]
        assert r2.states[2][3] == [0.25] * 8

    def test_vector_accessor(self):
        r = _make_result()
        assert r.vector(layer=2, position=3) == [0.25] * 8
        with pytest.raises(KeyError):
            r.vector(layer=1, position=3)


class TestPositionResolution:
    def test_last_input(self):
        assert _resolve_positions("last_input", n_input=7) == [6]

    def test_all_input(self):
        assert _resolve_positions("all_input", n_input=3) == [0, 1, 2]

    def test_explicit_list(self):
        assert _resolve_positions([0, 5], n_input=7) == [0, 5]

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _resolve_positions([7], n_input=7)
        with pytest.raises(ValueError):
            _resolve_positions([-1], n_input=7)

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError):
            _resolve_positions("everything", n_input=7)


class TestLayerResolution:
    def test_all(self):
        assert _resolve_layers("all", n_layers=4) == [0, 1, 2, 3]

    def test_explicit_list(self):
        assert _resolve_layers([1, 3], n_layers=4) == [1, 3]

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _resolve_layers([4], n_layers=4)

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError):
            _resolve_layers("final", n_layers=4)


@pytest.mark.skipif(not INTEGRATION, reason="set DRIFT_MLX_INTEGRATION=1 to run")
class TestCaptureIntegration:
    @pytest.fixture(scope="class")
    def loaded(self):
        pytest.importorskip("mlx.core")
        from mlx_lm import load

        return load(SMALL_MODEL)

    def test_prefill_capture_smoke(self, loaded):
        from drift.probing.mlx_hidden import _probe_hidden_with_model

        mlx_model, tokenizer = loaded
        n_layers = len(mlx_model.model.layers)
        hidden_dim = mlx_model.model.layers[0].mlp.down_proj.weight.shape[0] \
            if hasattr(mlx_model.model.layers[0], "mlp") else None

        result = _probe_hidden_with_model(
            mlx_model, tokenizer,
            system="You are a helpful assistant.",
            prompt="Say hello.",
            model_name=SMALL_MODEL,
            layers=[0, n_layers - 1],
            positions="last_input",
            prefill_only=True,
        )

        assert result.content is None  # prefill-only: nothing generated
        assert result.n_input > 0
        assert result.layers == [0, n_layers - 1]
        assert result.positions == [result.n_input - 1]

        vec = result.vector(n_layers - 1, result.n_input - 1)
        assert len(vec) > 0
        if hidden_dim:
            assert len(vec) == hidden_dim
        assert all(v == v for v in vec)  # no NaN
        assert any(v != 0.0 for v in vec)

    def test_generate_mode_returns_content(self, loaded):
        from drift.probing.mlx_hidden import _probe_hidden_with_model

        mlx_model, tokenizer = loaded
        result = _probe_hidden_with_model(
            mlx_model, tokenizer,
            system="You are a helpful assistant.",
            prompt="Say exactly: hi",
            model_name=SMALL_MODEL,
            layers=[0],
            positions="last_input",
            prefill_only=False,
            max_tokens=8,
        )
        assert result.content is not None
        assert len(result.content) > 0
