// Seedable PRNG using Mulberry32 algorithm
// Replaces Python's random.Random for reproducible games in the browser.

class SeededRNG {
  constructor(seed) {
    this._state = seed >>> 0; // ensure unsigned 32-bit
  }

  // Returns float in [0, 1)
  random() {
    this._state |= 0;
    this._state = (this._state + 0x6d2b79f5) | 0;
    let t = Math.imul(this._state ^ (this._state >>> 15), 1 | this._state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Returns integer in [min, max] inclusive
  randint(min, max) {
    return min + Math.floor(this.random() * (max - min + 1));
  }

  // Returns random element from array
  choice(arr) {
    if (arr.length === 0) throw new Error("Cannot choose from empty array");
    return arr[Math.floor(this.random() * arr.length)];
  }

  // Fisher-Yates in-place shuffle, returns the array
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
