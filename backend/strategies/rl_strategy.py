from __future__ import annotations

from pathlib import Path

from engine.game import Action, GameState
from strategies.base import Strategy


class RLStrategy(Strategy):
    """Inference wrapper for a trained DominoNet.

    Requires PyTorch:  pip install torch

    Usage:
        strategy = RLStrategy(model_path="models/domino_rl.pt")

    If ``model_path`` is None the network starts with random weights (useful
    for quick integration tests, not for competitive play).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        hidden_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from rl.encoding import decode_action, encode_action_mask, encode_state
            from rl.network import DominoNet
        except ImportError as exc:
            raise ImportError(
                "RLStrategy requires PyTorch.  Install with:  pip install torch"
            ) from exc

        self._torch = torch
        self._encode_state = encode_state
        self._encode_mask = encode_action_mask
        self._decode = decode_action
        self._device = torch.device(device)

        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dim = checkpoint.get("state_dim", 426)  # backward compat
                net = DominoNet(hidden_dim=hidden_dim, state_dim=state_dim)
                net.load_state_dict(checkpoint["model_state_dict"])
            else:
                net = DominoNet(hidden_dim=hidden_dim)
                net.load_state_dict(checkpoint)
        else:
            net = DominoNet(hidden_dim=hidden_dim)
        net.to(self._device)
        net.eval()
        self._net = net

    def choose_action(self, state: GameState) -> Action:
        s_t = self._encode_state(state).unsqueeze(0).to(self._device)
        m_t = self._encode_mask(state.valid_actions).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            probs, _ = self._net(s_t, m_t)

        action_idx = int(probs[0].argmax().item())
        return self._decode(action_idx, state.valid_actions)

    @property
    def name(self) -> str:
        return "RLStrategy"
