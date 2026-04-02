from __future__ import annotations

from pathlib import Path

import torch

from rl.encoding import ACTION_DIM, STATE_DIM
from rl.network import DominoNet


def export_to_onnx(model: DominoNet, output_path: str | Path) -> None:
    """Export a DominoNet to ONNX for browser inference via onnxruntime-web.

    Inputs:  state       float32 (B, STATE_DIM)
             action_mask bool    (B, ACTION_DIM)
    Outputs: action_probs float32 (B, ACTION_DIM)
             value        float32 (B, 1)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_state = torch.zeros(1, STATE_DIM)
    dummy_mask = torch.ones(1, ACTION_DIM, dtype=torch.bool)

    torch.onnx.export(
        model,
        (dummy_state, dummy_mask),
        str(output_path),
        input_names=["state", "action_mask"],
        output_names=["action_probs", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "action_mask": {0: "batch"},
            "action_probs": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported to {output_path}")
