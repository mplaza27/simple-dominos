// AI strategies — direct port of backend/strategies/
// Each strategy extends a base Strategy class and implements chooseAction(state).

// ── Base Strategy ──────────────────────────────────────────────

class Strategy {
  chooseAction(state) {
    throw new Error("chooseAction() must be implemented");
  }
  onGameStart() {}
  onGameEnd() {}
  get name() {
    return this.constructor.name;
  }
}

// ── HumanStrategy (stub — UI calls game.applyAction() directly) ─

class HumanStrategy extends Strategy {
  chooseAction(state) {
    throw new Error("HumanStrategy.chooseAction() should never be called — use game.applyAction()");
  }
  get name() { return "HumanStrategy"; }
}

// ── RandomStrategy ─────────────────────────────────────────────

class RandomStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }
  chooseAction(state) {
    return this._rng.choice(state.validActions);
  }
}

// ── GreedyStrategy ─────────────────────────────────────────────

class GreedyStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }
  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    const maxPips = Math.max(...playActions.map(a => a.tile.pipSum));
    const best = playActions.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(best);
  }
}

// ── NonGreedyStrategy ──────────────────────────────────────────

class NonGreedyStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }
  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    const minPips = Math.min(...playActions.map(a => a.tile.pipSum));
    const best = playActions.filter(a => a.tile.pipSum === minPips);
    return this._rng.choice(best);
  }
}

// ── GreedyDoublesStrategy ──────────────────────────────────────

class GreedyDoublesStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }
  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    // Prefer doubles
    const doubles = playActions.filter(a => a.tile.isDouble);
    const pool = doubles.length > 0 ? doubles : playActions;

    const maxPips = Math.max(...pool.map(a => a.tile.pipSum));
    const best = pool.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(best);
  }
}

// ── LateGameStrategy ───────────────────────────────────────────

class LateGameStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }

  _connectivity(state, action) {
    const tile = action.tile;
    let count = 0;
    for (const other of state.hand) {
      if (other === tile) continue;
      if (other.hasValue(tile.high) || other.hasValue(tile.low)) count++;
    }
    if (tile.isDouble) count += 2;
    return count;
  }

  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    // Play the tile with the LOWEST connectivity
    const minConn = Math.min(...playActions.map(a => this._connectivity(state, a)));
    const leastConnected = playActions.filter(a => this._connectivity(state, a) === minConn);

    // Among equally low connectivity, prefer higher pip sum
    const maxPips = Math.max(...leastConnected.map(a => a.tile.pipSum));
    const best = leastConnected.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(best);
  }
}

// ── NeverPassedStrategy ────────────────────────────────────────

class NeverPassedStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }

  _remainingCoverage(state, action) {
    const values = new Set();
    for (const tile of state.hand) {
      if (tile === action.tile) continue;
      values.add(tile.high);
      values.add(tile.low);
    }
    return values.size;
  }

  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    const maxCoverage = Math.max(...playActions.map(a => this._remainingCoverage(state, a)));
    const best = playActions.filter(a => this._remainingCoverage(state, a) === maxCoverage);

    // Tiebreak: highest pip sum
    const maxPips = Math.max(...best.map(a => a.tile.pipSum));
    const final = best.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(final);
  }
}

// ── PassTrackerStrategy ────────────────────────────────────────

class PassTrackerStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }

  _inferMissingValues(state) {
    const missing = {};
    // Replay the board from move history to know ends at each point
    const board = new Board();
    for (const [seat, action] of state.moveHistory) {
      if (action.isPass) {
        const ends = board.ends;
        if (ends !== null && seat !== state.currentSeat) {
          if (!missing[seat]) missing[seat] = new Set();
          missing[seat].add(ends[0]);
          missing[seat].add(ends[1]);
        }
      } else if (action.tile !== null && action.end !== null) {
        board.play(action.tile, action.end);
      }
    }
    return missing;
  }

  _pickHighestPip(actions) {
    const maxPips = Math.max(...actions.map(a => a.tile.pipSum));
    const best = actions.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(best);
  }

  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    const missing = this._inferMissingValues(state);
    if (Object.keys(missing).length === 0) {
      return this._pickHighestPip(playActions);
    }

    // All values opponents are known to lack
    const allMissing = new Set();
    for (const vals of Object.values(missing)) {
      for (const v of vals) allMissing.add(v);
    }

    function exploitScore(action) {
      const tile = action.tile;
      let newVals;
      if (state.boardEnds === null) {
        newVals = new Set([tile.high, tile.low]);
      } else if (action.end === "left") {
        newVals = new Set([tile.otherValue(state.boardEnds[0])]);
      } else {
        newVals = new Set([tile.otherValue(state.boardEnds[1])]);
      }
      let count = 0;
      for (const v of newVals) {
        if (allMissing.has(v)) count++;
      }
      return count;
    }

    const maxScore = Math.max(...playActions.map(a => exploitScore(a)));
    if (maxScore > 0) {
      const best = playActions.filter(a => exploitScore(a) === maxScore);
      return this._pickHighestPip(best);
    }

    return this._pickHighestPip(playActions);
  }
}

// ── PartnerAwareStrategy ───────────────────────────────────────

class PartnerAwareStrategy extends Strategy {
  constructor(rng) {
    super();
    this._rng = rng;
  }

  _analyzePartner(state) {
    const partnerValues = new Set();
    const partnerMissing = new Set();
    let partnerLastEnd = null;
    const teammate = state.teammateSeat;

    if (teammate === null) return { partnerValues, partnerMissing, partnerLastEnd };

    const board = new Board();
    for (const [seat, action] of state.moveHistory) {
      if (seat === teammate) {
        if (action.isPass) {
          const ends = board.ends;
          if (ends !== null) {
            partnerMissing.add(ends[0]);
            partnerMissing.add(ends[1]);
          }
        } else if (action.tile !== null && action.end !== null) {
          partnerValues.add(action.tile.high);
          partnerValues.add(action.tile.low);
          partnerLastEnd = action.end;
        }
      }
      if (!action.isPass && action.tile !== null && action.end !== null) {
        board.play(action.tile, action.end);
      }
    }

    return { partnerValues, partnerMissing, partnerLastEnd };
  }

  _pickHighestPip(actions) {
    const maxPips = Math.max(...actions.map(a => a.tile.pipSum));
    const best = actions.filter(a => a.tile.pipSum === maxPips);
    return this._rng.choice(best);
  }

  chooseAction(state) {
    const actions = state.validActions;
    if (actions.length === 1) return actions[0];

    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];

    // Non-pairs mode: fall back to greedy
    if (state.gameMode !== GameMode.PAIRS_4P || state.teammateSeat === null) {
      return this._pickHighestPip(playActions);
    }

    const { partnerValues, partnerMissing, partnerLastEnd } = this._analyzePartner(state);

    const score = (action) => {
      const tile = action.tile;
      let s = 0.0;

      // Reward playing tiles with values partner has shown
      for (const v of partnerValues) {
        if (tile.hasValue(v)) {
          s += 3.0;
          break;
        }
      }

      // Evaluate new exposed end
      if (state.boardEnds !== null) {
        let newEnd;
        if (action.end === "left") {
          newEnd = tile.otherValue(state.boardEnds[0]);
        } else {
          newEnd = tile.otherValue(state.boardEnds[1]);
        }

        if (partnerMissing.has(newEnd)) s -= 2.0;
        if (partnerValues.has(newEnd)) s += 2.0;
      }

      // Play opposite end from partner
      if (partnerLastEnd !== null) {
        const opposite = partnerLastEnd === "left" ? "right" : "left";
        if (action.end === opposite) s += 1.0;
      }

      // Small tiebreaker: higher pip sum
      s += tile.pipSum / 100.0;

      return s;
    };

    const maxScore = Math.max(...playActions.map(a => score(a)));
    const best = playActions.filter(a => score(a) === maxScore);
    return this._rng.choice(best);
  }
}

// ── Strategy Registry ──────────────────────────────────────────

const STRATEGY_REGISTRY = {
  RandomStrategy:        (rng) => new RandomStrategy(rng),
  GreedyStrategy:        (rng) => new GreedyStrategy(rng),
  NonGreedyStrategy:     (rng) => new NonGreedyStrategy(rng),
  GreedyDoublesStrategy: (rng) => new GreedyDoublesStrategy(rng),
  LateGameStrategy:      (rng) => new LateGameStrategy(rng),
  NeverPassedStrategy:   (rng) => new NeverPassedStrategy(rng),
  PassTrackerStrategy:   (rng) => new PassTrackerStrategy(rng),
  PartnerAwareStrategy:  (rng) => new PartnerAwareStrategy(rng),
};
