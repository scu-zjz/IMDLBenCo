import pytest
import torch
from unittest.mock import patch
from IMDLBenCo.evaluation import ImageF1, ImageAUC, ImageAccuracy  # 替换为你的实际路径

@pytest.fixture(autouse=True)
def patch_distributed_env():
    with patch("IMDLBenCo.training_scripts.utils.misc.get_world_size", return_value=1), \
         patch("IMDLBenCo.training_scripts.utils.misc.get_rank", return_value=0), \
         patch("torch.distributed.all_gather", side_effect=lambda out_list, tensor: out_list.__setitem__(0, tensor.clone())):
        yield

@pytest.mark.parametrize("EvaluatorClass, expected_value", [
    (ImageF1, torch.tensor(1.0)),
    (ImageAUC, torch.tensor(1.0)),
    (ImageAccuracy, torch.tensor(1.0)),
])
def test_epoch_update_with_empty_predict(EvaluatorClass, expected_value):
    evaluator = EvaluatorClass()
    evaluator.remain_update(torch.tensor([[0.8], [0.2]]), torch.tensor([[1.0], [0.0]]))
    result = evaluator.epoch_update()
    # 转换成 float 以兼容 float32 / float64 / Tensor 类型
    result = result.item() if isinstance(result, torch.Tensor) else float(result)
    assert result == pytest.approx(expected_value, abs=1e-5), f"{EvaluatorClass.__name__} failed on remain-only data"

@pytest.mark.parametrize("EvaluatorClass, expected_value", [
    (ImageF1, torch.tensor(1.0)),
    (ImageAUC, torch.tensor(1.0)),
    (ImageAccuracy, torch.tensor(1.0)),
])
def test_epoch_update_with_empty_remain(EvaluatorClass, expected_value):
    evaluator = EvaluatorClass()
    evaluator.batch_update(torch.tensor([[0.9], [0.1]]), torch.tensor([[1.0], [0.0]]))
    result = evaluator.epoch_update()
    result = torch.tensor(result, dtype=torch.float32) if not isinstance(result, torch.Tensor) else result
    assert torch.isclose(result, expected_value, atol=1e-5), f"{EvaluatorClass.__name__} mismatch with only batch data"

@pytest.mark.parametrize("EvaluatorClass", [ImageF1, ImageAUC, ImageAccuracy])
def test_epoch_update_with_both_empty(EvaluatorClass):
    evaluator = EvaluatorClass()
    with pytest.raises((RuntimeError)):
        _ = evaluator.epoch_update()
