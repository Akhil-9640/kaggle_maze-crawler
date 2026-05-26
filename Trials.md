# Trials Log: Maze Crawler Agent Development

This document details the development history, experimental trials, modifications, and benchmark results for our Maze Crawler agent.

---

## 1. Where We Started (The Baseline)
Our baseline agent (`main_backup.py`) was a rule-based agent with a core control loop of:
`Observation -> WorldModel -> Target Generation -> Strategy -> Action Commitment`

### Baseline Strengths & Weaknesses
* **Strengths:** Good basic routing, scrolling protection, and crystal harvesting.
* **Weaknesses:** 
  * Only won ~30% of games against the smart BFS agent (`starter_bfs.py`).
  * Factory suffered from horizontal oscillation (getting stuck moving East/West repeatedly when blocked to the North).
  * Economy was strictly passive (relying solely on crystals; miner functionality was disabled).
  * Hardcoded limits and independent builders sometimes led to cooldown blockages.

---

## 2. Experimental Branches & Agent Versions
We developed and compared several agent variations to optimize movement, building, and economic scaling.

### Version A: Throttling (`agent_throttle.py`)
* **Goal:** Reduce energy waste in the early game.
* **Logic:** Throttled factory movement and build frequencies when no immediate threats were present and energy was low.
* **Result:** Improved energy reserves slightly but made the agent too passive, sometimes leading to scrolling elimination in fast-scaling mid-games.

### Version B: Vanguard Coordination (`agent_vanguard.py`)
* **Goal:** Coordinate scouts and workers to proactively clear paths.
* **Logic:** Used path-linked targeting where scouts checked if a route was blocked and workers targeted specifically the blocking walls.
* **Result:** Highly effective at clearing paths but suffered from high energy consumption and worker collisions.

---

## 3. Iterative Tuning & Optimization (v2 to v7)

Using insights from the branches above, we developed a series of unified agents:

| Version | Key Changes & Tuning | Benchmark vs. BFS | Benchmark vs. Baseline | Status |
| :--- | :--- | :---: | :---: | :---: |
| **`agent_v2`** | Merged basic throttling and vanguard logic. | 40% Win | 50% Win | Abandoned |
| **`agent_v3`** | Added pathing adjustments for scouts to avoid dead ends. | 45% Win | 50% Win | Abandoned |
| **`agent_v4`** | Tuned worker building thresholds and wall removal priorities. | 45% Win | 55% Win | Abandoned |
| **`agent_v5`** | **The Winning Recipe:** Enabled miners, added anti-oscillation, unified build checks. | **50% Win** | **65% Win** | **Selected & Pushed** |
| **`agent_v6`** | Lowered factory worker energy reserve parameter to 300. | 25% Win | 50% Win | Failed (Degraded) |
| **`agent_v7`** | Tuned v5 with miner threshold tweaks. | 50% Win | 55% Win | Redundant (Identical to v5) |

---

## 4. Key Modifications in the Winning Agent (v5)

### A. Economic Expansion (`MINER_TARGET_COUNT = 1`)
We enabled miners to seek out mining nodes and transform into energy mines (generating 50 energy per turn). This gives us a massive passive engine to dominate the tiebreaker cascade (which factors in total team energy).

### B. Anti-Oscillation Movement Filter
We added history tracking to the factory. If the factory chooses a non-North direction (e.g. East or West) and that destination exists in its recent history (`history[-6:]`), the move is blocked. This forces the factory to:
1. Wait for a worker to clear the northern wall.
2. Build a worker if possible.
3. Perform a **JUMP** action to bypass the obstacle.

### C. Proactive Construction (`FACTORY_ROUTE_BLOCKS_BUILD = False`)
Instead of stopping construction when a clear route is found, the factory now builds units proactively, ensuring workers are ready to clear ahead before the factory gets blocked.

### D. Unified Build Manager
We rerouted worker-specific build functions through a centralized `factory_build` manager to prevent overlapping build cooldowns and ensure energy limits are respected.

---

## 5. Final Statistics & Validation

### Against `starter_bfs.py` (20 Games)
* **Win Rate:** **50.0%** (10 Wins, 7 Losses, 3 Draws)
* **Average Reward (Ours):** **191.93**
* **Average Reward (BFS):** **122.12**
* **Errors:** 0

### Against `main_backup.py` / Baseline (20 Games)
* **Win Rate:** **65.0%** (13 Wins, 6 Losses, 1 Draw)
* **Average Reward (Ours):** **451.68**
* **Average Reward (Baseline):** **-3.33** (Baseline frequently eliminated by scrolling)
* **Errors:** 0
