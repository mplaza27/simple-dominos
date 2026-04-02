from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from engine.game import Action, Game, GameState
from engine.player import Player
from engine.types import GameMode
from rl.encoding import STATE_DIM, decode_action, encode_action_mask, encode_state
from rl.network import DominoNet
from strategies.base import Strategy
from strategies.greedy_doubles_strategy import GreedyDoublesStrategy
from strategies.greedy_strategy import GreedyStrategy
from strategies.late_game_strategy import LateGameStrategy
from strategies.never_passed_strategy import NeverPassedStrategy
from strategies.non_greedy_strategy import NonGreedyStrategy
from strategies.partner_aware_strategy import PartnerAwareStrategy
from strategies.pass_tracker_strategy import PassTrackerStrategy
from strategies.random_strategy import RandomStrategy


@dataclass
class TrainingPhase:
    """Configuration for one phase of curriculum training."""
    name: str
    episodes: int  # how many episodes in this phase
    opponent_types: list[type[Strategy]]  # which opponents to use
    self_play_fraction: float  # 0.0 to 1.0
    self_play_update_interval: int  # how often to snapshot frozen net


DEFAULT_PHASES: list[TrainingPhase] = [
    TrainingPhase(
        name="curriculum",
        episodes=200_000,
        opponent_types=[RandomStrategy, NonGreedyStrategy, LateGameStrategy,
                        GreedyStrategy, GreedyDoublesStrategy,
                        PartnerAwareStrategy, PassTrackerStrategy,
                        NeverPassedStrategy],
        self_play_fraction=0.0,
        self_play_update_interval=5000,
    ),
    TrainingPhase(
        name="hard_opponents",
        episodes=300_000,
        opponent_types=[NeverPassedStrategy, PassTrackerStrategy,
                        GreedyDoublesStrategy, PartnerAwareStrategy],
        self_play_fraction=0.4,
        self_play_update_interval=3000,
    ),
    TrainingPhase(
        name="self_play",
        episodes=500_000,
        opponent_types=[NeverPassedStrategy, PassTrackerStrategy,
                        GreedyDoublesStrategy, PartnerAwareStrategy],
        self_play_fraction=0.7,
        self_play_update_interval=2000,
    ),
]


@dataclass
class _Step:
    state: torch.Tensor        # (STATE_DIM,) — saved for PPO recomputation
    action_mask: torch.Tensor  # (ACTION_DIM,) bool
    action_idx: int            # chosen action index
    log_prob: torch.Tensor     # old log-prob (detached)
    entropy: torch.Tensor
    value: torch.Tensor        # old value estimate (detached)


class _RLTrainStrategy(Strategy):
    """Strategy that samples from the network and records the trajectory."""

    def __init__(
        self,
        net: DominoNet,
        device: torch.device,
        record: bool = True,
    ) -> None:
        self._net = net
        self._device = device
        self._record = record
        self.steps: list[_Step] = []

    def choose_action(self, state: GameState) -> Action:
        s_t = encode_state(state).unsqueeze(0).to(self._device)
        m_t = encode_action_mask(state.valid_actions).unsqueeze(0).to(self._device)

        if self._record:
            probs, value = self._net(s_t, m_t)
            dist = Categorical(probs[0])
            action_idx = dist.sample()

            self.steps.append(
                _Step(
                    state=s_t.squeeze(0).detach(),
                    action_mask=m_t.squeeze(0).detach(),
                    action_idx=action_idx.item(),
                    log_prob=dist.log_prob(action_idx).detach(),
                    entropy=dist.entropy().detach(),
                    value=value[0, 0].detach(),
                )
            )
            return decode_action(action_idx.item(), state.valid_actions)
        else:
            # Frozen opponent: no gradient, no recording
            with torch.no_grad():
                probs, _ = self._net(s_t, m_t)
            action_idx = int(probs[0].argmax().item())
            return decode_action(action_idx, state.valid_actions)


@dataclass
class TrainStats:
    episode: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    win_rate: float = 0.0


_OPPONENT_STRATEGIES: list[type[Strategy]] = [
    RandomStrategy,
    GreedyStrategy,
    GreedyDoublesStrategy,
    PartnerAwareStrategy,
    NonGreedyStrategy,
    LateGameStrategy,
    NeverPassedStrategy,
    PassTrackerStrategy,
]


class SelfPlayTrainer:
    """PPO trainer with mixed opponents and periodic self-play.

    RL agent plays seats 0,2 (Team A) against rotating rule-based opponents
    (and occasionally a frozen copy of itself) on seats 1,3 (Team B).
    After ``batch_size`` games, PPO gradient updates are applied.
    """

    def __init__(
        self,
        net: DominoNet,
        device: torch.device | str = "cpu",
        lr: float = 3e-4,
        entropy_coef: float = 0.05,
        value_coef: float = 0.25,
        batch_size: int = 64,
        rng_seed: int = 0,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        self_play_fraction: float = 0.2,
        self_play_update_interval: int = 5000,
    ) -> None:
        self._net = net
        self._device = torch.device(device)
        self._net.to(self._device)
        self._optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._batch_size = batch_size
        self._rng = random.Random(rng_seed)
        self._opponent_idx = 0
        # PPO hyperparameters
        self._clip_eps = clip_eps
        self._ppo_epochs = ppo_epochs
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        # Self-play
        self._self_play_fraction = self_play_fraction
        self._self_play_update_interval = self_play_update_interval
        self._frozen_net: DominoNet | None = None
        self._episodes_since_freeze: int = 0
        self._total_episodes: int = 0
        # Active opponent pool (updated per phase in phased training)
        self._active_opponents: list[type[Strategy]] = list(_OPPONENT_STRATEGIES)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _maybe_update_frozen(self) -> None:
        """Snapshot current net weights for self-play if interval reached."""
        if self._episodes_since_freeze >= self._self_play_update_interval or self._frozen_net is None:
            self._frozen_net = copy.deepcopy(self._net)
            self._frozen_net.eval()
            self._episodes_since_freeze = 0

    def _run_game(self) -> list[tuple[list[_Step], float]]:
        """Run one mixed-opponent game.

        RL plays seats 0,2 (Team A); a rule-based strategy (or frozen self-play
        opponent) plays seats 1,3 (Team B).
        Returns [(steps, reward)] only for the two RL seats.
        """
        rl_strategies = {
            0: _RLTrainStrategy(self._net, self._device, record=True),
            2: _RLTrainStrategy(self._net, self._device, record=True),
        }

        # Decide opponent: self-play or rule-based
        use_self_play = (
            self._frozen_net is not None
            and self._rng.random() < self._self_play_fraction
        )

        if use_self_play:
            assert self._frozen_net is not None
            opp1: Strategy = _RLTrainStrategy(self._frozen_net, self._device, record=False)
            opp3: Strategy = _RLTrainStrategy(self._frozen_net, self._device, record=False)
        else:
            opponent_cls = self._active_opponents[self._opponent_idx % len(self._active_opponents)]
            self._opponent_idx += 1
            opp1 = opponent_cls()
            opp3 = opponent_cls()

        strategies: dict[int, Strategy] = {
            0: rl_strategies[0],
            1: opp1,
            2: rl_strategies[2],
            3: opp3,
        }
        players = [Player(seat=i, strategy=strategies[i]) for i in range(4)]

        # Record initial pip sums for RL seats (reward shaping)
        game = Game(
            players=players,
            game_mode=GameMode.PAIRS_4P,
            rng=random.Random(self._rng.randint(0, 2**63)),
        )
        initial_pips = {s: players[s].pip_sum for s in (0, 2)}

        result = game.play_round(max_moves=200, max_seconds=15)

        # Compute rewards for RL seats only
        out: list[tuple[list[_Step], float]] = []
        for s in (0, 2):
            # Base win/loss signal
            if result.winner_seats:
                base = 1.0 if s in result.winner_seats else -1.0
            else:
                base = 0.0

            # Pip-reduction shaping: reward for tiles played
            init_p = initial_pips[s]
            if init_p > 0:
                final_p = result.pip_sums.get(s, 0)
                shape = 0.1 * (init_p - final_p) / init_p
            else:
                shape = 0.0

            out.append((rl_strategies[s].steps, base + shape))
        return out

    def _ppo_update(
        self, batch: list[list[tuple[list[_Step], float]]]
    ) -> TrainStats:
        """PPO gradient update over a batch of games."""
        # ── 1. Compute GAE advantages per trajectory ──
        all_states: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_actions: list[int] = []
        all_old_log_probs: list[torch.Tensor] = []
        all_advantages: list[float] = []
        all_returns: list[float] = []

        wins = 0
        n_players = 0

        for game_episodes in batch:
            for steps, reward in game_episodes:
                if not steps:
                    if reward > 0:
                        wins += 1
                    n_players += 1
                    continue

                # GAE computation — walk backward
                # All intermediate rewards are 0; final step gets terminal reward
                T = len(steps)
                advantages = [0.0] * T
                returns = [0.0] * T

                gae = 0.0
                for t in reversed(range(T)):
                    r_t = reward if t == T - 1 else 0.0
                    v_t = steps[t].value.item()
                    v_next = steps[t + 1].value.item() if t + 1 < T else 0.0
                    delta = r_t + self._gamma * v_next - v_t
                    gae = delta + self._gamma * self._gae_lambda * gae
                    advantages[t] = gae
                    returns[t] = gae + v_t

                for t, step in enumerate(steps):
                    all_states.append(step.state)
                    all_masks.append(step.action_mask)
                    all_actions.append(step.action_idx)
                    all_old_log_probs.append(step.log_prob)
                    all_advantages.append(advantages[t])
                    all_returns.append(returns[t])

                if reward > 0:
                    wins += 1
                n_players += 1

        if not all_states:
            return TrainStats(win_rate=wins / max(n_players, 1))

        # ── 2. Flatten into tensors ──
        states_t = torch.stack(all_states).to(self._device)
        masks_t = torch.stack(all_masks).to(self._device)
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self._device)
        old_log_probs_t = torch.stack(all_old_log_probs).to(self._device)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=self._device)
        returns_t = torch.tensor(all_returns, dtype=torch.float32, device=self._device)

        # Normalize advantages
        if advantages_t.std() > 1e-6:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── 3. PPO epochs ──
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self._ppo_epochs):
            probs, values = self._net(states_t, masks_t)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns_t)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=0.5)
            self._optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n_epochs = self._ppo_epochs
        return TrainStats(
            policy_loss=total_policy_loss / n_epochs,
            value_loss=total_value_loss / n_epochs,
            entropy=total_entropy / n_epochs,
            win_rate=wins / max(n_players, 1),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def save_checkpoint(
        self, path: str, episode: int,
        phase_idx: int = 0, phase_episode: int = 0,
    ) -> None:
        """Save model weights, optimizer state, and episode/phase counts."""
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "episode": episode,
                "opponent_idx": self._opponent_idx,
                "state_dim": self._net.state_dim,
                "phase_idx": phase_idx,
                "phase_episode": phase_episode,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict:
        """Load model weights, optimizer state, and return checkpoint info.

        Returns a dict with keys: episode, phase_idx, phase_episode.
        For backward compat, callers that only need the episode count
        can use ``ckpt_info["episode"]``.
        """
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._net.load_state_dict(ckpt["model_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._opponent_idx = ckpt.get("opponent_idx", 0)
        return {
            "episode": ckpt["episode"],
            "phase_idx": ckpt.get("phase_idx", 0),
            "phase_episode": ckpt.get("phase_episode", 0),
        }

    @staticmethod
    def _curriculum_opponents(
        phase: TrainingPhase, episode_in_phase: int,
    ) -> list[type[Strategy]]:
        """For the 'curriculum' phase, gradually introduce opponents."""
        if phase.name != "curriculum":
            return phase.opponent_types
        total = phase.episodes
        n = len(phase.opponent_types)
        # 4 tiers across the phase duration
        tier_size = total // 4
        if episode_in_phase < tier_size:
            # Tier 1: first 3 opponents (Random, NonGreedy, LateGame)
            return phase.opponent_types[:3]
        elif episode_in_phase < 2 * tier_size:
            # Tier 2: + Greedy, GreedyDoubles
            return phase.opponent_types[:5]
        elif episode_in_phase < 3 * tier_size:
            # Tier 3: + PartnerAware, PassTracker
            return phase.opponent_types[:7]
        else:
            # Tier 4: all opponents including NeverPassed
            return phase.opponent_types[:n]

    def train_phased(
        self,
        phases: list[TrainingPhase],
        callback: Callable[[TrainStats], None] | None = None,
        start_episode: int = 0,
        start_phase_idx: int = 0,
        start_phase_episode: int = 0,
    ) -> None:
        """Train through multiple phases sequentially.

        Handles resume: if ``start_episode`` > 0, skips completed phases
        and resumes within the current one.
        """
        global_episode = start_episode

        for phase_idx, phase in enumerate(phases):
            if phase_idx < start_phase_idx:
                # Skip completed phases
                global_episode += phase.episodes - (
                    start_phase_episode if phase_idx == start_phase_idx else 0
                )
                continue

            # Determine how many episodes remain in this phase
            if phase_idx == start_phase_idx and start_phase_episode > 0:
                ep_in_phase_start = start_phase_episode
            else:
                ep_in_phase_start = 0

            remaining_in_phase = phase.episodes - ep_in_phase_start

            # Configure trainer for this phase
            self._self_play_fraction = phase.self_play_fraction
            self._self_play_update_interval = phase.self_play_update_interval

            print(f"\n{'='*60}")
            print(f"Phase {phase_idx + 1}/{len(phases)}: {phase.name}")
            print(f"  episodes: {phase.episodes}  "
                  f"self_play: {phase.self_play_fraction:.0%}  "
                  f"update_interval: {phase.self_play_update_interval}")
            if ep_in_phase_start > 0:
                print(f"  resuming from episode {ep_in_phase_start} in phase "
                      f"({remaining_in_phase} remaining)")
            print(f"{'='*60}\n")

            batch: list[list[tuple[list[_Step], float]]] = []

            for ep_offset in range(remaining_in_phase):
                ep_in_phase = ep_in_phase_start + ep_offset
                global_episode += 1
                self._total_episodes += 1
                self._episodes_since_freeze += 1
                self._maybe_update_frozen()

                # Update active opponents (curriculum graduation)
                self._active_opponents = self._curriculum_opponents(
                    phase, ep_in_phase,
                )
                self._opponent_idx = self._opponent_idx % max(len(self._active_opponents), 1)

                batch.append(self._run_game())

                if len(batch) >= self._batch_size:
                    stats = self._ppo_update(batch)
                    stats.episode = global_episode
                    batch = []
                    if callback:
                        callback(stats)

            # Flush any remaining partial batch
            if batch:
                stats = self._ppo_update(batch)
                stats.episode = global_episode
                if callback:
                    callback(stats)

    def train(
        self,
        num_episodes: int,
        callback: Callable[[TrainStats], None] | None = None,
        start_episode: int = 0,
    ) -> None:
        """Train for ``num_episodes`` games, calling ``callback`` after each update.

        If ``start_episode`` is set, episode numbering begins from that value
        (useful when resuming from a checkpoint).
        """
        batch: list[list[tuple[list[_Step], float]]] = []

        for episode in range(start_episode + 1, start_episode + num_episodes + 1):
            self._total_episodes += 1
            self._episodes_since_freeze += 1
            self._maybe_update_frozen()

            batch.append(self._run_game())

            if len(batch) >= self._batch_size:
                stats = self._ppo_update(batch)
                stats.episode = episode
                batch = []
                if callback:
                    callback(stats)

        # Flush any remaining partial batch
        if batch:
            stats = self._ppo_update(batch)
            stats.episode = start_episode + num_episodes
            if callback:
                callback(stats)
