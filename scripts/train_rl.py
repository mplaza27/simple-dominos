#!/usr/bin/env python3
"""Train the RL dominos agent via PPO with mixed opponents and self-play.

v2 adds phased curriculum training (default): 3 phases totalling 1M episodes
with graduated opponent introduction and increasing self-play fraction.

Quick start (v2 phased, 1M episodes):
    python scripts/train_rl.py

Legacy single-phase mode:
    python scripts/train_rl.py --no-phased --episodes 500000

Export to ONNX for browser inference:
    python scripts/train_rl.py --onnx-path frontend/public/models/domino_rl.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import torch

from rl.network import DominoNet
from rl.trainer import DEFAULT_PHASES, SelfPlayTrainer, TrainStats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL dominos agent (PPO, v2)")
    parser.add_argument("--episodes", type=int, default=500_000,
                        help="Total self-play games for single-phase mode (default: 500000)")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden layer width (default: 512)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of hidden layers in DominoNet (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Games per gradient update (default: 64)")
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy bonus coefficient (default: 0.05)")
    parser.add_argument("--value-coef", type=float, default=0.25,
                        help="Value loss coefficient (default: 0.25)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping epsilon (default: 0.2)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO optimization epochs per batch (default: 4)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for advantage estimation (default: 0.95)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--self-play-fraction", type=float, default=0.2,
                        help="Fraction of games against frozen self-play opponent (default: 0.2)")
    parser.add_argument("--self-play-update-interval", type=int, default=5000,
                        help="Episodes between frozen opponent weight snapshots (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device, e.g. cpu / cuda / mps (default: cpu)")
    parser.add_argument("--save-path", type=str, default="models/domino_rl.pt",
                        help="Where to save the trained model weights")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="If set, also export model to this ONNX path")
    parser.add_argument("--log-interval", type=int, default=500,
                        help="Print stats every N episodes (default: 500)")
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                        help="Save checkpoint every N episodes (default: 10000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the checkpoint at --save-path")
    parser.add_argument("--phased", action=argparse.BooleanOptionalAction, default=True,
                        help="Use v2 phased curriculum training (default: True). "
                             "Use --no-phased for legacy single-phase mode.")
    args = parser.parse_args()

    device = torch.device(args.device)
    net = DominoNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    trainer = SelfPlayTrainer(
        net=net,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        rng_seed=args.seed,
        clip_eps=args.clip_eps,
        ppo_epochs=args.ppo_epochs,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        self_play_fraction=args.self_play_fraction,
        self_play_update_interval=args.self_play_update_interval,
    )

    log_interval = args.log_interval
    checkpoint_interval = args.checkpoint_interval
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_episode = 0
    start_phase_idx = 0
    start_phase_episode = 0

    if args.resume and save_path.exists():
        ckpt_info = trainer.load_checkpoint(str(save_path))
        start_episode = ckpt_info["episode"]
        start_phase_idx = ckpt_info["phase_idx"]
        start_phase_episode = ckpt_info["phase_episode"]
        print(f"Resuming from checkpoint at episode {start_episode} "
              f"(phase {start_phase_idx}, phase_episode {start_phase_episode})")

    if args.phased:
        phases = DEFAULT_PHASES
        total_episodes = sum(p.episodes for p in phases)

        if args.episodes != 500_000:
            print(f"  Note: --episodes={args.episodes} is ignored in phased mode. "
                  f"Total episodes determined by phases: {total_episodes}")

        if start_episode >= total_episodes:
            print(f"Checkpoint at episode {start_episode} already >= {total_episodes}. Nothing to do.")
            return

        print(f"Training RL agent v2 (PPO, phased curriculum)")
        print(f"  total_episodes={total_episodes}  batch={args.batch_size}  "
              f"lr={args.lr}  hidden={args.hidden_dim}  "
              f"num_layers={args.num_layers}  "
              f"device={args.device}")
        print(f"  clip_eps={args.clip_eps}  ppo_epochs={args.ppo_epochs}  "
              f"gae_lambda={args.gae_lambda}  gamma={args.gamma}")
        print(f"  Phases:")
        for i, p in enumerate(phases):
            marker = " <-- resuming here" if (args.resume and i == start_phase_idx) else ""
            print(f"    {i+1}. {p.name}: {p.episodes} episodes, "
                  f"self_play={p.self_play_fraction:.0%}, "
                  f"update_interval={p.self_play_update_interval}{marker}")
        print()

        # Track phase info for checkpointing
        _current_phase_idx = start_phase_idx
        _current_phase_episode = start_phase_episode
        _phase_base_episode = start_episode - start_phase_episode

        def on_update_phased(stats: TrainStats) -> None:
            nonlocal _current_phase_idx, _current_phase_episode, _phase_base_episode
            # Compute current phase and episode within phase
            ep = stats.episode
            cumulative = 0
            for pi, phase in enumerate(phases):
                if ep <= cumulative + phase.episodes:
                    _current_phase_idx = pi
                    _current_phase_episode = ep - cumulative
                    break
                cumulative += phase.episodes

            if stats.episode % log_interval < args.batch_size:
                print(
                    f"  ep={stats.episode:7d}  "
                    f"policy={stats.policy_loss:+.4f}  "
                    f"value={stats.value_loss:.4f}  "
                    f"entropy={stats.entropy:.4f}  "
                    f"win_rate={stats.win_rate:.3f}"
                )
            if checkpoint_interval and stats.episode % checkpoint_interval < args.batch_size:
                trainer.save_checkpoint(
                    str(save_path), stats.episode,
                    phase_idx=_current_phase_idx,
                    phase_episode=_current_phase_episode,
                )
                print(f"    [checkpoint saved to {save_path} at ep {stats.episode} "
                      f"(phase {_current_phase_idx}:{_current_phase_episode})]")

        trainer.train_phased(
            phases=phases,
            callback=on_update_phased,
            start_episode=start_episode,
            start_phase_idx=start_phase_idx,
            start_phase_episode=start_phase_episode,
        )

        trainer.save_checkpoint(
            str(save_path), total_episodes,
            phase_idx=len(phases) - 1,
            phase_episode=phases[-1].episodes,
        )
        print(f"\nModel saved to {save_path}")

    else:
        # Legacy single-phase mode
        remaining = args.episodes - start_episode
        if remaining <= 0:
            print(f"Checkpoint at episode {start_episode} already >= {args.episodes}. Nothing to do.")
            return

        print(f"Training RL agent (PPO, mixed opponents + self-play)")
        print(f"  episodes={args.episodes}  batch={args.batch_size}  "
              f"lr={args.lr}  hidden={args.hidden_dim}  "
              f"num_layers={args.num_layers}  "
              f"entropy={args.entropy_coef}  value={args.value_coef}  "
              f"device={args.device}")
        print(f"  clip_eps={args.clip_eps}  ppo_epochs={args.ppo_epochs}  "
              f"gae_lambda={args.gae_lambda}  gamma={args.gamma}")
        print(f"  self_play_fraction={args.self_play_fraction}  "
              f"self_play_update_interval={args.self_play_update_interval}")
        if start_episode > 0:
            print(f"  resuming from episode {start_episode}")
        print()

        def on_update(stats: TrainStats) -> None:
            if stats.episode % log_interval < args.batch_size:
                print(
                    f"  ep={stats.episode:7d}  "
                    f"policy={stats.policy_loss:+.4f}  "
                    f"value={stats.value_loss:.4f}  "
                    f"entropy={stats.entropy:.4f}  "
                    f"win_rate={stats.win_rate:.3f}"
                )
            if checkpoint_interval and stats.episode % checkpoint_interval < args.batch_size:
                trainer.save_checkpoint(str(save_path), stats.episode)
                print(f"    [checkpoint saved to {save_path} at ep {stats.episode}]")

        trainer.train(num_episodes=remaining, callback=on_update, start_episode=start_episode)

        trainer.save_checkpoint(str(save_path), args.episodes)
        print(f"\nModel saved to {save_path}")

    if args.onnx_path:
        from rl.export import export_to_onnx
        export_to_onnx(net, args.onnx_path)


if __name__ == "__main__":
    main()
