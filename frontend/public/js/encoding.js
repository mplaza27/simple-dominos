// State encoding for RL model — direct port of backend/rl/encoding.py
// Encodes a JS GameState into a 511-dim Float32Array for ONNX inference.

const STATE_DIM = 511;
const ACTION_DIM = 111;
const PASS_IDX = 110;
const _MAX_TURNS = 50;

function tileToIdx(tile) {
  return tile.high * (tile.high + 1) / 2 + tile.low;
}

function actionToIdx(action) {
  if (action.isPass) return PASS_IDX;
  return tileToIdx(action.tile) * 2 + (action.end === 'left' ? 0 : 1);
}

function encodeState(state) {
  const vec = new Float32Array(STATE_DIM);
  let offset = 0;

  // 1. Hand tiles (55-dim binary)
  for (const tile of state.hand) {
    vec[offset + tileToIdx(tile)] = 1.0;
  }
  offset += 55;

  // 2. Played tiles (55-dim binary)
  for (const tile of state.playedTiles) {
    vec[offset + tileToIdx(tile)] = 1.0;
  }
  offset += 55;

  // 3-4. Board ends one-hot (10-dim each), 5. Board empty flag
  if (state.boardEnds === null) {
    vec[offset + 20] = 1.0; // empty flag at offset+20
  } else {
    vec[offset + state.boardEnds[0]] = 1.0;       // left end one-hot
    vec[offset + 10 + state.boardEnds[1]] = 1.0;  // right end one-hot
  }
  offset += 21;

  // 6. Opponent tile counts normalized (3-dim)
  for (let i = 0; i < 3 && i < state.opponentTileCounts.length; i++) {
    vec[offset + i] = state.opponentTileCounts[i] / 10.0;
  }
  offset += 3;

  // 7. Pass history per seat (4-dim)
  for (let i = 0; i < 4 && i < state.passHistory.length; i++) {
    vec[offset + i] = state.passHistory[i] ? 1.0 : 0.0;
  }
  offset += 4;

  // 8. Is first move (1-dim)
  vec[offset] = state.isFirstMove ? 1.0 : 0.0;
  offset += 1;

  // 9. Current seat one-hot (4-dim)
  if (state.currentSeat >= 0 && state.currentSeat < 4) {
    vec[offset + state.currentSeat] = 1.0;
  }
  offset += 4;

  // 10. Teammate seat one-hot (4-dim)
  if (state.teammateSeat !== null && state.teammateSeat >= 0 && state.teammateSeat < 4) {
    vec[offset + state.teammateSeat] = 1.0;
  }
  offset += 4;

  // 11-13. Walk move_history to derive:
  //   - Per-seat played tiles (4 x 55 = 220 dims)
  //   - Cumulative pass count per seat (4 dims)
  //   - Tile play order (55 dims)
  //   Also track missing values for opponent inference

  const seatPlayedOffset = offset; // [offset .. offset+220)
  offset += 220;
  const passCountOffset = offset;  // [offset .. offset+4)
  offset += 4;
  const playOrderOffset = offset;  // [offset .. offset+55)
  offset += 55;

  const missingValues = {}; // seat -> Set of pip values
  const board = new Board();

  for (let turnIdx = 0; turnIdx < state.moveHistory.length; turnIdx++) {
    const [seat, action] = state.moveHistory[turnIdx];
    if (action.isPass) {
      if (seat >= 0 && seat < 4) {
        vec[passCountOffset + seat] += 1.0;
      }
      const ends = board.ends;
      if (ends !== null && seat !== state.currentSeat) {
        if (!missingValues[seat]) missingValues[seat] = new Set();
        missingValues[seat].add(ends[0]);
        missingValues[seat].add(ends[1]);
      }
    } else if (action.tile !== null) {
      const idx = tileToIdx(action.tile);
      if (seat >= 0 && seat < 4) {
        vec[seatPlayedOffset + seat * 55 + idx] = 1.0;
      }
      vec[playOrderOffset + idx] = (turnIdx + 1) / _MAX_TURNS;
      if (action.end !== null) {
        board.play(action.tile, action.end);
      }
    }
  }
  // Normalize pass counts
  for (let i = 0; i < 4; i++) {
    vec[passCountOffset + i] /= 10.0;
  }

  // 14. Inferred missing pip values per other seat (3 x 10 = 30 dims)
  const cur = state.currentSeat;
  for (let offsetIdx = 0; offsetIdx < 3; offsetIdx++) {
    const otherSeat = (cur + offsetIdx + 1) % 4;
    const missing = missingValues[otherSeat];
    if (missing) {
      for (const pipVal of missing) {
        if (pipVal >= 0 && pipVal <= 9) {
          vec[offset + offsetIdx * 10 + pipVal] = 1.0;
        }
      }
    }
  }
  offset += 30;

  // 15. Opponent threat estimate (3 dims)
  for (let i = 0; i < 3; i++) {
    const otherSeat = (cur + i + 1) % 4;
    const missing = missingValues[otherSeat];
    const nMissing = missing ? Math.min(missing.size, 10) : 0;
    vec[offset + i] = 1.0 - nMissing / 10.0;
  }
  offset += 3;

  // 16. Pip value counts (10 dims): fraction of played tiles containing each pip 0-9
  for (const tile of state.playedTiles) {
    vec[offset + tile.high] += 1.0;
    vec[offset + tile.low] += 1.0;
    if (tile.high === tile.low) {
      vec[offset + tile.high] -= 1.0; // doubles counted once
    }
  }
  for (let i = 0; i < 10; i++) {
    vec[offset + i] /= 10.0;
  }
  offset += 10;

  // 17. End scarcity (2 dims)
  if (state.boardEnds !== null) {
    for (let side = 0; side < 2; side++) {
      const endVal = state.boardEnds[side];
      let count = 0;
      for (const tile of state.playedTiles) {
        if (tile.high === endVal || tile.low === endVal) count++;
      }
      vec[offset + side] = count / 10.0;
    }
  }
  offset += 2;

  // 18. Per-seat pip value play counts (4 x 10 = 40 dims)
  for (const [seat, action] of state.moveHistory) {
    if (!action.isPass && action.tile !== null && seat >= 0 && seat < 4) {
      vec[offset + seat * 10 + action.tile.high] += 1.0;
      vec[offset + seat * 10 + action.tile.low] += 1.0;
      if (action.tile.high === action.tile.low) {
        vec[offset + seat * 10 + action.tile.high] -= 1.0;
      }
    }
  }
  for (let i = 0; i < 40; i++) {
    vec[offset + i] /= 10.0;
  }
  offset += 40;

  return vec;
}

function encodeActionMask(validActions) {
  const mask = new Float32Array(ACTION_DIM); // 0.0 = invalid, 1.0 = valid
  for (const action of validActions) {
    mask[actionToIdx(action)] = 1.0;
  }
  return mask;
}
