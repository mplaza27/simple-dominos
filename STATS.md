# Cuban Dominos (Double-9, 4P Pairs) — Statistics & Analysis

## 1. The Tile Set

A Double-9 set contains **55 unique tiles** — every unordered pair (a, b) where 0 <= a <= b <= 9.

| Property | Value |
|----------|-------|
| Total tiles | 55 = C(10,2) + 10 |
| Doubles | 10 (one per pip: 0-0 through 9-9) |
| Non-doubles | 45 |
| Total pip sum of set | 495 |
| Average pip sum per tile | 9.0 (exact) |
| Pip sum range | 0 (0-0) to 18 (9-9) |
| Pip sum std dev | 4.28 |

**Tiles per pip value:** Every pip value 0-9 appears on exactly **10 tiles**. For pip value V, the tiles are: (V,V), (V,V+1), ..., (V,9) and (0,V), (1,V), ..., (V-1,V). This always totals 10.

**Pip sum distribution** (symmetric, bell-shaped around 9):

| Sum | Tiles | Sum | Tiles |
|-----|-------|-----|-------|
| 0 | 1 | 10 | 5 |
| 1 | 1 | 11 | 4 |
| 2 | 2 | 12 | 4 |
| 3 | 2 | 13 | 3 |
| 4 | 3 | 14 | 3 |
| 5 | 3 | 15 | 2 |
| 6 | 4 | 16 | 2 |
| 7 | 4 | 17 | 1 |
| 8 | 5 | 18 | 1 |
| 9 | 5 | | |

---

## 2. The Deal

4 players each receive 10 tiles. 15 tiles remain hidden and out of play for the entire round.

**Total distinct deals:**
```
55! / ((10!)^4 * 15!) = ~5.6 x 10^34
```

### Hand Probabilities

| Statistic | Value |
|-----------|-------|
| P(specific tile in your hand) | 10/55 = **2/11 = 18.2%** |
| P(specific tile in hidden 15) | 15/55 = **3/11 = 27.3%** |
| P(specific tile in a specific opponent's hand) | 10/55 = **2/11 = 18.2%** |
| Expected pip sum of a hand | **90** (exact) |
| Std dev of hand pip sum | **12.25** (variance = 150) |
| Expected doubles in hand | **20/11 = 1.82** |

### Doubles in Hand Distribution

| Doubles in hand | Probability |
|-----------------|-------------|
| 0 | 10.9% |
| 1 | 30.3% |
| **2** | **33.2%** (most likely) |
| 3 | 18.6% |
| 4 | 5.8% |
| 5 | 1.1% |
| 6+ | 0.1% |

---

## 3. First Player Determination

The player holding the highest double opens. If all doubles of a given rank are in the hidden 15, the next-lower double is checked.

P(a specific double is in someone's hand) = 40/55 = **8/11 = 72.7%**
P(a specific double is in the hidden 15) = 15/55 = **3/11 = 27.3%**

### Which Double Opens?

| Double | P(determines opener) | Cumulative |
|--------|---------------------|------------|
| **9-9** | **72.73%** | 72.73% |
| 8-8 | 20.20% | 92.93% |
| 7-7 | 5.34% | 98.27% |
| 6-6 | 1.33% | 99.60% |
| 5-5 | 0.31% | 99.91% |
| 4-4 | 0.07% | 99.98% |
| 3-3 | 0.01% | ~100% |
| 2-2 or lower | <0.01% | ~100% |
| No doubles in play | 1 in 9.7 million | 100% |

Formula: P(highest double = k) = (8/11) * (3/11)^(9-k)

**The 9-9 opens nearly 3 out of 4 games.** 99.6% of games are determined by the 6-6 or higher.

---

## 4. Hidden Information

### Bits of Hidden Information

| Perspective | Hidden arrangements | Bits |
|-------------|-------------------|------|
| Single player (45 unseen tiles) | 45! / (10! * 10! * 10! * 15!) | **~80.7 bits** |
| God's-eye (full deal) | 55! / (10!^4 * 15!) | **~115.4 bits** |

### Comparison to Other Games

| Game | Hidden info (per player) | Hidden info (total) |
|------|------------------------|---------------------|
| **Double-9 Dominos 4P** | **~80.7 bits** | **~115.4 bits** |
| Contract Bridge | ~56.2 bits | ~95.4 bits |
| Texas Hold'em Poker | ~10.4 bits | ~10.4 bits per street |
| Chess | 0 bits | 0 bits |

**Double-9 partnership dominos has more hidden information than bridge** — a game widely regarded as the gold standard of partnership card games.

### The 15-Tile Fog of War

The 15 undealt tiles create **permanent, irreducible uncertainty**. Unlike bridge where all 52 cards are dealt and perfect counting can eventually deduce every hand, in dominos you can never distinguish "opponent doesn't have tile X" from "tile X is in the hidden pile."

- **27.3% of the tile set** is permanently unknown
- Expected tiles of any given pip value in the hidden pile: **2.73 out of 10**
- P(all 10 tiles of a pip value are hidden): ~1 in 10 million (effectively impossible)

**Distribution of hidden tiles per pip value** (hypergeometric):

| Hidden count | Probability |
|-------------|-------------|
| 0 | 2.9% |
| 1 | 14.0% |
| **2** | **27.6%** |
| **3** | **29.0%** (most likely) |
| 4 | 17.9% |
| 5 | 6.8% |
| 6+ | 1.8% |

Most often, 2-3 tiles of each pip value are hidden. This means for any pip value, you're typically working with 7-8 of the 10 tiles distributed among the four players.

---

## 5. Branching Factor & Pass Probability

The board has two open ends with values L and R. A tile is playable if it contains L or R.

- **L != R (common case):** 19 tiles in the set match (10 containing L + 10 containing R - 1 overlap tile [L|R])
- **L == R (rare case):** 10 tiles match

### Expected Valid Plays by Hand Size

| Hand size | Avg valid plays | P(must pass) | Game phase |
|-----------|----------------|-------------|------------|
| **10** | **4.9** | **1.9%** | Opening |
| 9 | 4.1 | 3.0% | Early |
| 8 | 3.6 | 4.4% | Early |
| 7 | 2.8 | 7.6% | Mid |
| 6 | 2.3 | 11.7% | Mid |
| **5** | **1.5** | **20.9%** | Late |
| 4 | 1.3 | 26.4% | Late |
| 3 | 0.9 | 37.3% | End |
| 2 | 0.6 | 50.5% | End |
| **1** | **0.25** | **76.0%** | Final |

**Overall average branching factor: ~2.0 actions per turn** (including pass as an action).

Key insight: **at hand size 10, you almost always have a choice** (98% of the time, 4-5 options). But choices narrow fast — by hand size 3, you're passing over a third of the time. The game's strategic decisions are front-loaded.

---

## 6. Game Length & Stalemates

Based on 10,000 simulated games with random play:

| Statistic | Value |
|-----------|-------|
| Average game length | **41.3 moves** (stdev 5.9) |
| Range | 14-65 moves |
| Games ending in empty hand (domino) | **39.6%** |
| **Games ending in stalemate** | **60.4%** |
| Stalemate ties (both teams share minimum) | 3.0% of stalemates |
| Average remaining tiles per player at end | 2.03 |

### Stalemate Mechanics

A stalemate (locked board / "tranque") occurs when all 4 players pass consecutively — both open ends show values nobody can match.

For the board to lock with ends showing X and Y (X != Y):
- 19 tiles in the set contain X or Y
- ALL remaining unplayed tiles containing X or Y must be in the hidden 15
- Since at most 15 tiles are hidden, at least 4 of the 19 X/Y tiles must have already been played

**With random play, stalemates are the majority outcome (60%).** Skilled play reduces this because:
- Players actively avoid creating lockable board states (unless they want a lock)
- Better hand management means more players can domino out
- Teams can coordinate to keep board ends on values they hold

**With skilled play, stalemates are estimated at ~20-30%** — they become a deliberate tactical choice rather than an accident.

---

## 7. Passing Statistics

From 10,000 random-play games:

| Statistic | Value |
|-----------|-------|
| Average passes per game | **9.4** |
| Average passes per player | **2.3** |
| Pass rate at hand size 10 | 1.9% |
| Pass rate at hand size 5 | 20.9% |
| Pass rate at hand size 1 | 76.0% |

### Information Value of a Pass

A pass is one of the most informative events in the game. When a player passes with board ends showing X and Y:
- They hold **zero** tiles containing X AND zero tiles containing Y
- This eliminates up to 19 tiles from their possible hand
- Late-game passes (small hand) are even more constraining — if they have 2 tiles and pass on ends 5 and 7, neither tile contains a 5 or 7

**A pass often reveals more information than a play.** Playing a tile tells you one specific tile they had. Passing tells you about potentially dozens of tiles they don't have.

---

## 8. Scoring

Points awarded = opposing team's remaining pip sum when a team wins.

| Statistic | Value |
|-----------|-------|
| Average points per round | **42.8** |
| Median | 40 |
| Std dev | 21.7 |
| Range observed | 0 to 182 |
| Theoretical maximum | 270 (opponents hold all high-sum tiles) |

A score of 0 occurs in stalemate ties (both teams have a player at the minimum pip sum).

---

## 9. Skill vs. Luck

### Where Skill Lives

| Skill dimension | Description |
|----------------|-------------|
| **Tile counting** | Tracking which of the 10 tiles per pip value have been played, by whom |
| **Pass inference** | Deducing opponent holdings from what they passed on |
| **Board control** | Managing open ends to force opponents to pass or play undesirably |
| **Lock strategy** | Deliberately creating or avoiding stalemates based on pip advantage |
| **Partner signaling** | Communicating hand strength through play choices (no talking allowed) |
| **End-game counting** | Exact deduction of remaining tiles through elimination |
| **Hand management** | Balancing pip reduction vs. maintaining connectivity across values |

### Estimated Win Rates

| Matchup | Win rate (stronger team) |
|---------|------------------------|
| Expert pair vs. random play | ~65-75% |
| Expert pair vs. intermediate | ~60-65% |
| Expert pair vs. competent pair | ~55-60% |

For comparison: poker pros win ~60-80% heads-up vs amateurs; bridge experts win ~60-65% vs intermediates.

### What Limits Skill Expression

1. **The deal is random** — sometimes you get 10 tiles with no doubles and no matching values; no amount of skill fixes a terrible hand
2. **15 hidden tiles** create irreducible uncertainty (~27% of the set is permanently unknown)
3. **Low branching factor** (avg ~2 actions/turn) means many decisions are forced
4. **Partnership coordination** depends on both players being skilled — one weak partner drags the team down
5. **Short rounds** (~41 moves) mean high variance per round; skill emerges over many rounds

### The "No Draw" Rule

Cuban dominos uses block rules (no boneyard). This **increases skill** relative to draw variants:
- In draw dominos, you draw random tiles until you can play — luck dominates
- In block dominos, passing reveals information, hand management matters, and avoiding passes is a core skill
- The block rule makes pip counting and inference far more valuable

---

## 10. Comparison: Double-6 vs. Double-9

| Metric | Double-6 (2P) | Double-9 (4P Pairs) | Factor |
|--------|---------------|---------------------|--------|
| Tiles in set | 28 | 55 | 2.0x |
| Tiles per hand | 7 | 10 | 1.4x |
| Players | 2 | 4 | 2x |
| Teams | none | 2 | partnership adds signaling |
| Hidden info (bits) | ~37 | ~115 | ~3.1x |
| Pip values | 7 (0-6) | 10 (0-9) | 1.4x |
| Tiles per pip value | 7 | 10 | 1.4x |
| Total pip sum | 168 | 495 | 2.9x |
| Game tree (est.) | ~2^107 | ~2^185 | ~10^23x more complex |

Double-9 4P Pairs is **astronomically more complex** — roughly 10^23 times larger game tree. The combination of more tiles, partnership dynamics, and a wider pip range makes it one of the most complex domino variants played competitively.

---

## 11. Key Formulas

### Tiles containing pip value V
```
count(V) = 10    (for all V in 0..9)
```

### Total pip appearances for value V
```
appearances(V) = 11    (the double contributes 2, the other 9 tiles contribute 1 each)
```

### Probability of being dealt a specific tile
```
P = 10/55 = 2/11 = 0.1818
```

### Expected doubles in hand
```
E[doubles] = 10 * (10/55) = 20/11 = 1.818
```

### P(zero doubles in hand)
```
P = C(45,10) / C(55,10) = 0.1091
```

### Expected pip sum of hand
```
E[sum] = (10/55) * 495 = 90
```

### Variance of hand pip sum
```
Var = n * S^2/N * (N-n)/(N-1)
    where n=10, N=55, S^2 = sum of (pip_i - 9)^2 / 55
    = 150
Std = sqrt(150) = 12.25
```

### P(highest double in play = k)
```
P(k) = (40/55) * (15/55)^(9-k)    for k = 0..9
     = (8/11) * (3/11)^(9-k)
```

### Expected playable tiles (hand size n, board ends L != R)
```
E[playable] = n * 19/55 = n * 0.3455
```

### P(forced pass) with hand size n (board ends L != R)
```
P(pass) = C(36, n) / C(55, n)    (none of n tiles are among the 19 matching)
         ≈ ((55-19)/55)^n for large sets
```

---

*All simulation statistics based on 10,000 random-play games. Exact combinatorial values computed analytically. See PRD.md for game rules.*
