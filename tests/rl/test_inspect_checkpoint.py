import numpy as np
import pytest

from experiments.rl.inspect_rl_checkpoint import _json_default


def test_checkpoint_inspector_serializes_numpy_metadata():
    assert _json_default(np.asarray((1, 2))) == [1, 2]
    assert _json_default(np.float32(0.5)) == pytest.approx(0.5)


def test_checkpoint_inspector_rejects_unknown_metadata_type():
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())
