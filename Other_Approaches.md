# Other Potential Approaches for Maze Crawler

This document outlines fundamentally different architectural and algorithmic approaches that could yield better performance or higher win rates against advanced opponents.

---

## 1. Centralized Swarm Coordination (Task Assignment)
* **Concept:** Instead of each robot independently calculating its closest target (which causes redundant paths, resource competition, and potential collisions), treat path assignment as a global optimization problem.
* **Implementation:**
  * Use the **Hungarian Algorithm** (Linear Sum Assignment) or **Min-Cost Max-Flow** on every turn.
  * Construct a cost matrix where rows are friendly robots and columns are targets (crystals, mining nodes, blocking walls, charging mines).
  * Compute pairwise path distances (using A* or BFS) as the costs.
  * Globally solve the matrix to assign a unique, optimal target to each robot.
* **Benefits:** Prevents friendly units from targeting the same resources, maximizes collection throughput, and prevents accidental friendly-fire collisions.

---

## 2. Dynamic Influence Fields (Potential Fields)
* **Concept:** Navigate the grid using mathematical fields rather than static discrete pathfinding.
* **Implementation:**
  * Generate a 2D scalar field over the 20x20 grid.
  * **Attractors (Positive Field):** Crystals, mining nodes, and friendly mines.
  * **Repellers (Negative Field):** The advancing southern scroll boundary (exponential penalty near the edge), enemy units (based on crush hierarchy), and dead ends.
  * At any turn, a robot moves to the adjacent cell with the highest net potential.
* **Benefits:** Seamlessly balances competing goals (e.g., escaping the scroll zone vs. gathering nearby energy) dynamically, without complex nested if-else rules.

---

## 3. Aggressive Sabotage (Worker Corridor Blocking)
* **Concept:** Actively weaponize the scrolling boundary against the enemy factory.
* **Implementation:**
  * Since the board is symmetric with doors connecting both sides, send a worker across the mirror axis.
  * Identify tight corridors or bottlenecks directly in front of the enemy factory.
  * Use `BUILD_DIR` actions to place walls, blocking the enemy's path northward.
  * Because scroll speeds ramp up rapidly in the late game (1 row per turn), trapping or delaying the enemy factory for just a few turns is a guaranteed win.

---

## 4. Zero-Sum Local Minimax Search
* **Concept:** Run game-theoretic search when units are close to the opponent.
* **Implementation:**
  * Trigger a local minimax search (3–4 steps deep) when the enemy factory or enemy combat units enter a small window (e.g., 5x5) around your units.
  * Model possible move/build options for both sides.
  * Solve for the best minimax strategy, specifically optimizing for when to use the Factory's **JUMP** action (20-turn cooldown) to crush key enemy units or avoid being trapped.

---

## 5. Active Node Denial (Economic Blockades)
* **Concept:** Deny the enemy passive mining income.
* **Implementation:**
  * Mines generate 50 energy per turn (up to 1000). A player with uncontested mines easily wins energy tiebreakers.
  * Send fast Scouts (move period 1) across the mirror axis doors to park on mining nodes on the enemy's half of the map.
  * An enemy miner cannot transform while a scout occupies the node. They must bring a higher-tier unit to crush the scout, wasting valuable turns and energy.

---

## 6. Hybrid Reinforcement Learning
* **Concept:** Combine rule-based low-level mechanics with neural network-based high-level strategy.
* **Implementation:**
  * Retain deterministic pathfinders, collision checkers, and boundary safety loops in Python code.
  * Use a policy network (trained via PPO or Q-learning in self-play) to determine global state decisions: e.g., "Build miner now," "Play defensively," "Initiate sabotage rush."
