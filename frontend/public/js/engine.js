// Game engine — direct port of backend/engine/ (types.py, tile.py, player.py, game.py)

// ── Types ──────────────────────────────────────────────────────

const GameMode = Object.freeze({
  FFA_2P: "ffa_2p",
  FFA_3P: "ffa_3p",
  PAIRS_4P: "pairs_4p",
});

const PLAYER_COUNTS = Object.freeze({
  [GameMode.FFA_2P]: 2,
  [GameMode.FFA_3P]: 3,
  [GameMode.PAIRS_4P]: 4,
});

const TILES_PER_PLAYER = 10;

class InvalidMoveError extends Error {
  constructor(msg) {
    super(msg);
    this.name = "InvalidMoveError";
  }
}

// ── Tile (interned, immutable) ─────────────────────────────────

const _tileCache = new Map();

class Tile {
  constructor(high, low) {
    this.high = high;
    this.low = low;
    Object.freeze(this);
  }

  // Factory with interning — ensures === works for identity comparison
  static of(a, b) {
    const hi = a >= b ? a : b;
    const lo = a >= b ? b : a;
    const key = hi * 10 + lo;
    let t = _tileCache.get(key);
    if (!t) {
      t = new Tile(hi, lo);
      _tileCache.set(key, t);
    }
    return t;
  }

  get pipSum() {
    return this.high + this.low;
  }

  get isDouble() {
    return this.high === this.low;
  }

  hasValue(value) {
    return this.high === value || this.low === value;
  }

  otherValue(value) {
    if (this.high === value) return this.low;
    if (this.low === value) return this.high;
    throw new Error(`Tile ${this} does not have value ${value}`);
  }

  equals(other) {
    return other instanceof Tile && this.high === other.high && this.low === other.low;
  }

  toString() {
    return `[${this.low}|${this.high}]`;
  }
}

// ── Board (mutable) ────────────────────────────────────────────

class Board {
  constructor() {
    this._leftEnd = null;
    this._rightEnd = null;
    this._tiles = [];
  }

  get isEmpty() {
    return this._leftEnd === null;
  }

  get ends() {
    if (this.isEmpty) return null;
    return [this._leftEnd, this._rightEnd];
  }

  get playedTiles() {
    return new Set(this._tiles);
  }

  get tileList() {
    return this._tiles.slice();
  }

  canPlay(tile) {
    if (this.isEmpty) return true;
    return tile.hasValue(this._leftEnd) || tile.hasValue(this._rightEnd);
  }

  validPlacements(tile) {
    if (this.isEmpty) return ["left"];
    const placements = [];
    if (tile.hasValue(this._leftEnd)) {
      placements.push("left");
    }
    if (tile.hasValue(this._rightEnd)) {
      if (this._leftEnd !== this._rightEnd || placements.length === 0) {
        placements.push("right");
      } else if (tile.isDouble) {
        placements.push("right");
      }
    }
    return placements;
  }

  play(tile, end) {
    if (this.isEmpty) {
      this._leftEnd = tile.low;
      this._rightEnd = tile.high;
      this._tiles.push(tile);
      return;
    }

    if (end === "left") {
      if (!tile.hasValue(this._leftEnd)) {
        throw new InvalidMoveError(
          `Tile ${tile} cannot be played on left end ${this._leftEnd}`
        );
      }
      this._leftEnd = tile.otherValue(this._leftEnd);
      this._tiles.push(tile);
    } else if (end === "right") {
      if (!tile.hasValue(this._rightEnd)) {
        throw new InvalidMoveError(
          `Tile ${tile} cannot be played on right end ${this._rightEnd}`
        );
      }
      this._rightEnd = tile.otherValue(this._rightEnd);
      this._tiles.push(tile);
    } else {
      throw new InvalidMoveError(`Invalid end: ${end}`);
    }
  }
}

// ── Action (immutable) ─────────────────────────────────────────

class Action {
  constructor(tile, end) {
    this.tile = tile;   // Tile | null
    this.end = end;     // "left" | "right" | null
    Object.freeze(this);
  }

  static pass() {
    return new Action(null, null);
  }

  static play(tile, end) {
    return new Action(tile, end);
  }

  get isPass() {
    return this.tile === null;
  }

  equals(other) {
    if (!(other instanceof Action)) return false;
    if (this.isPass && other.isPass) return true;
    if (this.isPass || other.isPass) return false;
    return this.tile === other.tile && this.end === other.end;
  }

  toString() {
    if (this.isPass) return "Action(PASS)";
    return `Action(${this.tile} -> ${this.end})`;
  }
}

// ── Player (mutable hand) ──────────────────────────────────────

class Player {
  constructor(seat, strategy, name) {
    this.seat = seat;
    this.strategy = strategy;
    this.name = name || "";
    this.hand = [];
  }

  get tileCount() {
    return this.hand.length;
  }

  get pipSum() {
    let sum = 0;
    for (const t of this.hand) sum += t.pipSum;
    return sum;
  }

  get hasTiles() {
    return this.hand.length > 0;
  }

  removeTile(tile) {
    const idx = this.hand.indexOf(tile);
    if (idx === -1) throw new Error(`Tile ${tile} not in hand`);
    this.hand.splice(idx, 1);
  }

  playableTiles(leftEnd, rightEnd) {
    return this.hand.filter(t => t.hasValue(leftEnd) || t.hasValue(rightEnd));
  }
}

// ── GameState (immutable snapshot for strategies) ──────────────

class GameState {
  constructor({
    hand, boardEnds, playedTiles, opponentTileCounts,
    passHistory, validActions, currentSeat, teammateSeat,
    gameMode, isFirstMove, moveHistory, numPlayers,
  }) {
    this.hand = hand;                       // Tile[]
    this.boardEnds = boardEnds;             // [int, int] | null
    this.playedTiles = playedTiles;         // Set<Tile>
    this.opponentTileCounts = opponentTileCounts; // int[]
    this.passHistory = passHistory;         // bool[]
    this.validActions = validActions;       // Action[]
    this.currentSeat = currentSeat;         // int
    this.teammateSeat = teammateSeat;       // int | null
    this.gameMode = gameMode;               // GameMode value
    this.isFirstMove = isFirstMove;         // bool
    this.moveHistory = moveHistory;         // [seat, Action][]
    this.numPlayers = numPlayers;           // int
    Object.freeze(this);
  }
}

// ── RoundResult ────────────────────────────────────────────────

class RoundResult {
  constructor({ winnerSeats, isStalemate, finalHands, pipSums, points, moveHistory }) {
    this.winnerSeats = winnerSeats;   // int[]
    this.isStalemate = isStalemate;   // bool
    this.finalHands = finalHands;     // {seat: Tile[]}
    this.pipSums = pipSums;           // {seat: int}
    this.points = points;             // int
    this.moveHistory = moveHistory;   // [seat, Action][]
    Object.freeze(this);
  }
}

// ── Factory ────────────────────────────────────────────────────

function createFullSet() {
  const tiles = [];
  for (let i = 0; i < 10; i++) {
    for (let j = i; j < 10; j++) {
      tiles.push(Tile.of(i, j));
    }
  }
  return tiles;
}

// ── Game ───────────────────────────────────────────────────────

class Game {
  constructor(players, gameMode, rng) {
    const expected = PLAYER_COUNTS[gameMode];
    if (players.length !== expected) {
      throw new Error(
        `${gameMode} requires ${expected} players, got ${players.length}`
      );
    }

    this._players = players;
    this._gameMode = gameMode;
    this._rng = rng || new SeededRNG(Date.now());
    this._board = new Board();
    this._moveHistory = [];
    this._consecutivePasses = 0;
    this._gameOver = false;

    this._deal();
    this._currentIndex = this._determineFirstPlayer();
    this._isFirstMove = true;
  }

  _deal() {
    const tiles = createFullSet();
    this._rng.shuffle(tiles);
    let offset = 0;
    for (const player of this._players) {
      player.hand = tiles.slice(offset, offset + TILES_PER_PLAYER);
      offset += TILES_PER_PLAYER;
    }
    // Remaining tiles are out of play
  }

  _determineFirstPlayer() {
    // Find highest double across all hands
    let bestDouble = null;
    let bestSeat = 0;
    for (const player of this._players) {
      for (const tile of player.hand) {
        if (tile.isDouble) {
          if (bestDouble === null || tile.pipSum > bestDouble.pipSum) {
            bestDouble = tile;
            bestSeat = player.seat;
          }
        }
      }
    }
    if (bestDouble !== null) {
      this._startingTile = bestDouble;
      return bestSeat;
    }

    // No doubles: highest pip-sum tile
    let bestTile = null;
    bestSeat = 0;
    for (const player of this._players) {
      for (const tile of player.hand) {
        if (bestTile === null || tile.pipSum > bestTile.pipSum) {
          bestTile = tile;
          bestSeat = player.seat;
        } else if (tile.pipSum === bestTile.pipSum && tile.high > bestTile.high) {
          bestTile = tile;
          bestSeat = player.seat;
        }
      }
    }
    this._startingTile = bestTile;
    return bestSeat;
  }

  get currentPlayer() {
    return this._players[this._currentIndex];
  }

  get gameOver() {
    return this._gameOver;
  }

  get board() {
    return this._board;
  }

  get moveHistory() {
    return this._moveHistory;
  }

  _computeValidActions(player) {
    if (this._board.isEmpty) {
      // First move: must play the starting tile (highest double / highest pip-sum)
      const st = this._startingTile;
      const match = player.hand.find(t => t.low === st.low && t.high === st.high);
      if (match) return [Action.play(match, "left")];
      return player.hand.map(tile => Action.play(tile, "left"));
    }

    const actions = [];
    for (const tile of player.hand) {
      const placements = this._board.validPlacements(tile);
      for (const end of placements) {
        actions.push(Action.play(tile, end));
      }
    }

    if (actions.length === 0) {
      actions.push(Action.pass());
    }

    return actions;
  }

  _buildGameState(player) {
    const validActions = this._computeValidActions(player);

    const opponentCounts = [];
    for (const p of this._players) {
      if (p.seat !== player.seat) opponentCounts.push(p.tileCount);
    }

    // Pass history: track last action per player
    const numPlayers = this._players.length;
    const lastAction = {};
    for (let i = this._moveHistory.length - 1; i >= 0; i--) {
      const [seat, action] = this._moveHistory[i];
      if (!(seat in lastAction)) {
        lastAction[seat] = action;
      }
      if (Object.keys(lastAction).length === numPlayers) break;
    }
    const passHist = [];
    for (let i = 0; i < numPlayers; i++) {
      const la = lastAction[i];
      passHist.push(la ? la.isPass : false);
    }

    let teammate = null;
    if (this._gameMode === GameMode.PAIRS_4P) {
      teammate = (player.seat + 2) % 4;
    }

    return new GameState({
      hand: player.hand.slice(),
      boardEnds: this._board.ends,
      playedTiles: this._board.playedTiles,
      opponentTileCounts: opponentCounts,
      passHistory: passHist,
      validActions: validActions,
      currentSeat: player.seat,
      teammateSeat: teammate,
      gameMode: this._gameMode,
      isFirstMove: this._isFirstMove,
      moveHistory: this._moveHistory.slice(),
      numPlayers: numPlayers,
    });
  }

  _validateAndApply(player, action) {
    const validActions = this._computeValidActions(player);

    if (!validActions.some(a => a.equals(action))) {
      throw new InvalidMoveError(
        `Action ${action} is not valid. Valid actions: ${validActions}`
      );
    }

    if (action.isPass) {
      this._consecutivePasses++;
      this._moveHistory.push([player.seat, action]);
      return;
    }

    this._board.play(action.tile, action.end);
    player.removeTile(action.tile);
    this._consecutivePasses = 0;
    this._moveHistory.push([player.seat, action]);
    this._isFirstMove = false;
  }

  _advanceTurn() {
    this._currentIndex = (this._currentIndex + 1) % this._players.length;
  }

  _checkGameOver() {
    // Win: current player emptied hand
    if (!this.currentPlayer.hasTiles) return true;
    // Stalemate: all players passed consecutively
    if (this._consecutivePasses >= this._players.length) return true;
    return false;
  }

  // AI turn — calls strategy internally
  step() {
    if (this._gameOver) throw new Error("Game is already over");

    const player = this.currentPlayer;
    const state = this._buildGameState(player);
    const action = player.strategy.chooseAction(state);
    this._validateAndApply(player, action);

    const seat = player.seat;

    if (this._checkGameOver()) {
      this._gameOver = true;
      return { seat, action, done: true };
    }

    this._advanceTurn();
    return { seat, action, done: false };
  }

  // Expose state for human turn
  getState() {
    return this._buildGameState(this.currentPlayer);
  }

  // Apply human-chosen action
  applyAction(action) {
    if (this._gameOver) throw new Error("Game is already over");

    const player = this.currentPlayer;
    this._validateAndApply(player, action);

    const seat = player.seat;

    if (this._checkGameOver()) {
      this._gameOver = true;
      return { seat, action, done: true };
    }

    this._advanceTurn();
    return { seat, action, done: false };
  }

  playRound() {
    for (const p of this._players) {
      if (p.strategy.onGameStart) p.strategy.onGameStart();
    }
    while (!this._gameOver) {
      this.step();
    }
    for (const p of this._players) {
      if (p.strategy.onGameEnd) p.strategy.onGameEnd();
    }
    return this.buildResult();
  }

  buildResult() {
    const finalHands = {};
    const pipSums = {};
    for (const p of this._players) {
      finalHands[p.seat] = p.hand.slice();
      pipSums[p.seat] = p.pipSum;
    }

    // Check for empty-hand winner
    const emptyHandWinners = this._players
      .filter(p => !p.hasTiles)
      .map(p => p.seat);

    if (emptyHandWinners.length > 0) {
      if (this._gameMode === GameMode.PAIRS_4P) {
        const winnerSeat = emptyHandWinners[0];
        const winnerTeam = new Set([winnerSeat, (winnerSeat + 2) % 4]);
        const allSeats = new Set([0, 1, 2, 3]);
        const loserTeam = new Set([...allSeats].filter(s => !winnerTeam.has(s)));
        let points = 0;
        for (const s of loserTeam) points += pipSums[s];
        return new RoundResult({
          winnerSeats: [...winnerTeam].sort(),
          isStalemate: false,
          finalHands,
          pipSums,
          points,
          moveHistory: this._moveHistory.slice(),
        });
      } else {
        return new RoundResult({
          winnerSeats: emptyHandWinners,
          isStalemate: false,
          finalHands,
          pipSums,
          points: 0,
          moveHistory: this._moveHistory.slice(),
        });
      }
    }

    // Stalemate
    if (this._gameMode === GameMode.PAIRS_4P) {
      const teamA = new Set([0, 2]);
      const teamB = new Set([1, 3]);

      const minPips = Math.min(...Object.values(pipSums));
      const lowestSeats = Object.keys(pipSums)
        .map(Number)
        .filter(s => pipSums[s] === minPips);

      const aHasLowest = lowestSeats.some(s => teamA.has(s));
      const bHasLowest = lowestSeats.some(s => teamB.has(s));

      let winnerSeats, points;
      if (aHasLowest && !bHasLowest) {
        winnerSeats = [0, 2];
        points = pipSums[1] + pipSums[3];
      } else if (bHasLowest && !aHasLowest) {
        winnerSeats = [1, 3];
        points = pipSums[0] + pipSums[2];
      } else {
        winnerSeats = [];
        points = 0;
      }

      return new RoundResult({
        winnerSeats,
        isStalemate: true,
        finalHands,
        pipSums,
        points,
        moveHistory: this._moveHistory.slice(),
      });
    } else {
      const minPips = Math.min(...Object.values(pipSums));
      const winnerSeats = Object.keys(pipSums)
        .map(Number)
        .filter(s => pipSums[s] === minPips)
        .sort();

      return new RoundResult({
        winnerSeats,
        isStalemate: true,
        finalHands,
        pipSums,
        points: 0,
        moveHistory: this._moveHistory.slice(),
      });
    }
  }
}
