import torch

from IMDLBenCo.model_zoo import APSCNet
from IMDLBenCo.registry import MODELS


def test_apsc_net_is_registered_and_follows_output_contract():
    assert MODELS.get("APSCNet") is APSCNet

    # A one-block-per-stage backbone keeps the CI test small while exercising
    # the released decoder and the standard IMDLBenCo output contract.
    model = APSCNet(
        backbone_depths=(1, 1, 1, 1),
        drop_path_rate=0.0,
    ).eval()
    image = torch.randn(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)

    with torch.inference_mode():
        output = model(image=image, mask=mask, label=torch.zeros(1))

    assert output["pred_mask"].shape == (1, 1, 64, 64)
    assert output["pred_label"].shape == (1,)
    assert torch.isfinite(output["backward_loss"])
    assert torch.all((output["pred_mask"] >= 0) & (output["pred_mask"] <= 1))


def test_apsc_checkpoint_prefix_normalization():
    tensor = torch.ones(1)
    checkpoint = {"state_dict": {"module.model.weight": tensor}}
    state_dict = APSCNet._unwrap_state_dict(checkpoint)

    assert list(state_dict) == ["weight"]
    assert state_dict["weight"] is tensor
