// RL Strategy — loads ONNX model and runs inference in browser
// Requires: onnxruntime-web (ort), encoding.js, engine.js

let _rlSession = null;
let _rlLoadPromise = null;
let _rlLoadError = null;

async function loadRLModel() {
  if (_rlSession) return _rlSession;
  if (_rlLoadError) throw _rlLoadError;
  if (_rlLoadPromise) return _rlLoadPromise;

  _rlLoadPromise = (async () => {
    try {
      _rlSession = await ort.InferenceSession.create('models/domino_rl_v2.onnx', {
        executionProviders: ['wasm'],
      });
      console.log('RL model loaded successfully');
      return _rlSession;
    } catch (e) {
      _rlLoadError = e;
      console.error('Failed to load RL model:', e);
      throw e;
    }
  })();
  return _rlLoadPromise;
}

class RLStrategy extends Strategy {
  constructor() {
    super();
    this._ready = false;
    this._fallback = null;
  }

  async _ensureModel() {
    if (this._ready) return true;
    try {
      await loadRLModel();
      this._ready = true;
      return true;
    } catch (e) {
      return false;
    }
  }

  chooseAction(state) {
    // Synchronous wrapper — model should already be loaded by game start
    if (!this._ready || !_rlSession) {
      // Fallback to greedy if model isn't loaded
      console.warn('RL model not loaded, falling back to Greedy');
      return this._greedyFallback(state);
    }

    const stateVec = encodeState(state);
    const maskVec = encodeActionMask(state.validActions);

    // Run inference synchronously via pre-loaded session
    // ONNX Runtime Web supports sync inference for small models
    const stateTensor = new ort.Tensor('float32', stateVec, [1, STATE_DIM]);

    // The model expects bool mask but ONNX Runtime Web handles float->bool
    // Use bool tensor
    const maskBool = new Uint8Array(ACTION_DIM);
    for (let i = 0; i < ACTION_DIM; i++) {
      maskBool[i] = maskVec[i] > 0 ? 1 : 0;
    }
    const maskTensor = new ort.Tensor('bool', maskBool, [1, ACTION_DIM]);

    // We need async inference but chooseAction is sync.
    // Store the result from the last async call, or use a blocking approach.
    // Since ONNX Runtime Web's run() is async, we'll use a workaround:
    // pre-compute the action asynchronously and cache it.
    // For now, use the greedy fallback and queue async computation.
    // Actually, let's restructure: we'll override the game loop to handle async.

    // Return a placeholder — the actual integration uses chooseActionAsync
    return this._greedyFallback(state);
  }

  async chooseActionAsync(state) {
    if (!this._ready || !_rlSession) {
      const loaded = await this._ensureModel();
      if (!loaded) return this._greedyFallback(state);
    }

    const stateVec = encodeState(state);
    const maskVec = encodeActionMask(state.validActions);

    const stateTensor = new ort.Tensor('float32', stateVec, [1, STATE_DIM]);
    const maskBool = new Uint8Array(ACTION_DIM);
    for (let i = 0; i < ACTION_DIM; i++) {
      maskBool[i] = maskVec[i] > 0 ? 1 : 0;
    }
    const maskTensor = new ort.Tensor('bool', maskBool, [1, ACTION_DIM]);

    const results = await _rlSession.run({
      state: stateTensor,
      action_mask: maskTensor,
    });

    const probs = results.action_probs.data;

    // Pick the valid action with highest probability
    let bestIdx = -1;
    let bestProb = -1;
    for (const action of state.validActions) {
      const idx = actionToIdx(action);
      if (probs[idx] > bestProb) {
        bestProb = probs[idx];
        bestIdx = idx;
      }
    }

    // Map back to Action object
    for (const action of state.validActions) {
      if (actionToIdx(action) === bestIdx) {
        return action;
      }
    }

    // Should never happen — fallback
    return state.validActions[0];
  }

  _greedyFallback(state) {
    const actions = state.validActions;
    if (actions.length <= 1) return actions[0];
    const playActions = actions.filter(a => a.tile !== null);
    if (playActions.length === 0) return actions[0];
    let best = playActions[0];
    for (const a of playActions) {
      if (a.tile.pipSum > best.tile.pipSum) best = a;
    }
    return best;
  }

  get name() { return 'RLStrategy'; }
}

// Register in the global strategy registry
if (typeof STRATEGY_REGISTRY !== 'undefined') {
  STRATEGY_REGISTRY.RLStrategy = (rng) => new RLStrategy();
}
