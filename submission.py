"""Layered rule-based Maze Crawler agent.

The control stack is intentionally small and explicit:

    observation -> WorldModel -> candidate targets -> Strategy -> commitment

The strategy layer is the main place to tune behavior. Mechanics, memory,
pathing, cooldown checks, transform-vacating behavior, transfer accounting, and
reservation stay deterministic so experiments do not need to relearn the game
rules.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


# ============================================================
# Shared Setup
# ============================================================

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
TYPE_NAMES = {
    FACTORY: "factory",
    SCOUT: "scout",
    WORKER: "worker",
    MINER: "miner",
}

WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
DIRS = {
    "NORTH": (0, 1, WALL_N),
    "EAST": (1, 0, WALL_E),
    "WEST": (-1, 0, WALL_W),
    "SOUTH": (0, -1, WALL_S),
}
DIR_ORDER = ("NORTH", "EAST", "WEST", "SOUTH")
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}

ACTION_SET = {
    "IDLE",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "BUILD_SCOUT",
    "BUILD_WORKER",
    "BUILD_MINER",
    "BUILD_NORTH",
    "BUILD_SOUTH",
    "BUILD_EAST",
    "BUILD_WEST",
    "REMOVE_NORTH",
    "REMOVE_SOUTH",
    "REMOVE_EAST",
    "REMOVE_WEST",
    "TRANSFORM",
    "TRANSFER_NORTH",
    "TRANSFER_SOUTH",
    "TRANSFER_EAST",
    "TRANSFER_WEST",
    "JUMP_NORTH",
    "JUMP_SOUTH",
    "JUMP_EAST",
    "JUMP_WEST",
}

VISION_BY_TYPE = {
    FACTORY: 4,
    SCOUT: 5,
    WORKER: 3,
    MINER: 3,
}
MAX_ENERGY_BY_TYPE = {
    SCOUT: 100,
    WORKER: 300,
    MINER: 500,
}
COMBAT_RANK = {
    SCOUT: 1,
    WORKER: 2,
    MINER: 3,
    FACTORY: 4,
}

# Search budgets are deliberately bounded. If a target is too expensive to
# reach under the turn timeout, the policy should degrade to local safe motion
# instead of missing the action deadline.
MAX_BFS_NODES = 650
MAX_TARGETS_PER_KIND = 20
SOFT_ACT_DEADLINE = 0.72
MIN_TIME_REMAINING = 0.05

# The constants below are policy assumptions, not hidden rules. Keeping them
# centralized makes later paired-seed sweeps easier: change one knob, compare
# the same seeds, and inspect which failure metrics move.
SCROLL_DANGER_GAP = 3
SCROLL_CAUTION_GAP = 5
SCROLL_DANGER_RAMP_BONUS = 4
FACTORY_SACRIFICE_GAP = 1
WORKER_TARGET_COUNT = 1
SCOUT_TARGET_COUNT_OPENING = 1
SCOUT_TARGET_COUNT_MID = 1
MINER_TARGET_COUNT = 1
ALLOW_EXTRA_SCOUT = True
PROFILE = "baseline_route_floor_hybrid"
MIN_NODE_SCROLL_GAP = 6
MIN_MINE_LIFETIME_TURNS = 8
UNATTENDED_MINE_MIN_LIFETIME_TURNS = 14
MINE_HARVEST_MIN_ENERGY = 150
MINE_HARVEST_SCROLL_GAP = 4
UNLOAD_FRACTION = 0.75
SCOUT_RETURN_ENERGY = 75
SCOUT_FORCE_RETURN_ENERGY = 95

FACTORY_WORKER_RESERVE = 0
FACTORY_SCOUT_RESERVE = 160
FACTORY_MINER_RESERVE = 180
FACTORY_GENERAL_RESERVE = 120

CRYSTAL_VALUE_WEIGHT = 0.60
CRYSTAL_DIST_PENALTY = 6.0
FRONTIER_NORTH_BONUS = 0.55
NODE_NORTH_BONUS = 0.25
MINE_TRANSFORM_BUFFER = 20
MINE_TRANSFORM_LOST_UNIT_PENALTY = 20.0
WALL_REMOVE_BUFFER = 0
LOW_ENERGY_TRANSFER_THRESHOLD = 8
ASSIGNMENT_TTL = 12
ASSIGNMENT_SWITCH_MARGIN = 80.0
FACTORY_MOVE_FIRST_GAP = 999
OPENING_SCOUT_STEP_LIMIT = 120
FACTORY_BFS_LOOKAHEAD = 20
FACTORY_OPTIMISTIC_BFS_NODES = 900
FACTORY_JUMP_AWARE_BFS_NODES = 900
WORKER_VANGUARD_GAP = 5
WORKER_VANGUARD_AHEAD = 5
WORKER_VANGUARD_SCORE = 1500.0
WORKER_REFILL_ENERGY = 140
WORKER_REFILL_SAFE_GAP = 8
WORKER_REFILL_MAX_CRYSTAL_DIST = 8
WORKER_REFILL_CRYSTAL_BONUS = 120.0
OPENING_WORKER_BRANCH_EXIT_LIMIT = 1
JUMP_ON_NORTH_WALL_GAP = 3
JUMP_ON_NORTH_WALL_AFTER_STEP = 999
FACTORY_ROUTE_BLOCKS_BUILD = False
ENABLE_MARGIN_SURPLUS_SCOUT = True
SCOUT_BUILD_MARGIN_SURPLUS = 12
ENABLE_FACTORY_WORKER_REFUEL = True
FACTORY_WORKER_REFUEL_THRESHOLD = 150
FACTORY_WORKER_REFUEL_MAX_FACTORY_ENERGY = 450
FACTORY_WORKER_REFUEL_MAX_OVERFLOW = 250
FACTORY_WORKER_REFUEL_PRESSURE_OVERFLOW = 1000
FACTORY_FLOOR_PRESSURE_SURPLUS = 3
EMERGENCY_JUMP_IGNORES_MOVE_CD = True
MIN_MINER_BUILD_SCORE = 300.0
MINER_BUILD_MARGIN = 8

# Kaggle keeps module globals alive across turns. The default store is keyed by
# player for normal submissions; replay conversion and policy experiments can inject
# an isolated state_store through compute_actions().
STATE_BY_PLAYER = {}


@dataclass(frozen=True)
class Robot:
    uid: str
    rtype: int
    col: int
    row: int
    energy: int
    owner: int
    move_cd: int = 0
    jump_cd: int = 0
    build_cd: int = 0

    @property
    def pos(self):
        return (self.col, self.row)


@dataclass
class Target:
    """High-level intent candidate before primitive action commitment.

    Policy code ranks these targets, then the planner converts the chosen
    target into a primitive action with BFS, legality, and reservation checks.
    """

    kind: str
    pos: tuple[int, int]
    score: float


@dataclass(frozen=True)
class RouteQuality:
    """Factory route summary used to price tempo before building units."""

    action: str | None
    next_pos: tuple[int, int] | None
    exists: bool
    uses_jump: bool
    north_gain: int
    margin_surplus: int
    blocked: bool


@dataclass(frozen=True)
class SafetyDecision:
    """Pure factory-destination evaluation result.

    Factory movement can require tactical side effects: a blocker may be forced
    to vacate, or a transforming miner may free its cell before movement. The
    evaluation is kept separate from application so speculative candidate checks
    cannot leak reservations into the final plan.
    """

    ok: bool
    forced_actions: dict
    forced_reserved_next: dict
    reserved_next: set
    sacrifice_uids: set
    reason: str = ""


def to_plain_dict(obj):
    """Convert Struct-like objects into Python primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain_dict(v) for v in obj]
    if hasattr(obj, "items"):
        try:
            return {str(k): to_plain_dict(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            str(k): to_plain_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return obj


def cfg(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_pos_key(key):
    try:
        col, row = str(key).split(",", 1)
        return int(col), int(row)
    except Exception:
        return None


def manhattan(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def action_ready(cd):
    # Cooldowns tick before action validation.
    return int(cd or 0) <= 1


REQUIRE_MOVE_READY_FOR_JUMP = True


def dynamic_danger_gap_for_step(step, config):
    ramp = int(cfg(config, "scrollRampSteps", 400))
    progress = min(1.0, max(0.0, float(step) / max(1, ramp)))
    return SCROLL_DANGER_GAP + int(SCROLL_DANGER_RAMP_BONUS * progress)


def scroll_interval_for_step(step, config):
    start = int(cfg(config, "scrollStartInterval", 4))
    end = int(cfg(config, "scrollEndInterval", 1))
    ramp = int(cfg(config, "scrollRampSteps", 400))
    if step >= ramp:
        return end
    progress = step / max(1, ramp)
    interval = start - (start - end) * progress
    return max(end, round(interval))


def infer_south_bound_from_step(step, config):
    """Reconstruct hidden southBound when a per-player replay obs omits it."""
    south, _ = infer_scroll_state_from_step(step, config)
    return south


def infer_scroll_state_from_step(step, config):
    south = 0
    counter = int(cfg(config, "scrollStartInterval", 4))
    for past_step in range(max(0, int(step))):
        counter -= 1
        if counter <= 0:
            south += 1
            counter = scroll_interval_for_step(past_step, config)
    return south, max(1, counter)


def robot_record(uid, data):
    values = list(data) if isinstance(data, (list, tuple)) else []
    values += [0] * max(0, 8 - len(values))
    return Robot(
        uid=str(uid),
        rtype=int(values[0]),
        col=int(values[1]),
        row=int(values[2]),
        energy=int(values[3]),
        owner=int(values[4]),
        move_cd=int(values[5]),
        jump_cd=int(values[6]),
        build_cd=int(values[7]),
    )


def wants_to_unload(robot):
    cap = MAX_ENERGY_BY_TYPE.get(robot.rtype)
    return cap is not None and robot.energy >= cap * UNLOAD_FRACTION


def parse_mine_record(record):
    if isinstance(record, (list, tuple)):
        values = list(record) + [0, 0, None, 0]
        return int(values[0]), int(values[1]), values[2], int(values[3])
    return 0, 0, None, 0


def estimated_mine_energy(world, mine_pos, mine_data):
    energy, max_energy, owner, last_seen = parse_mine_record(mine_data)
    try:
        owner = int(owner)
    except Exception:
        return energy
    if owner != world.player:
        return energy

    cap = max_energy if max_energy > 0 else energy
    if mine_pos in world.visible_cells:
        return min(cap, energy)

    # Mines are remembered by the observation after discovery. Outside vision
    # the record can be stale, so friendly mine energy is projected forward
    # from the last turn on which the mine was actually visible.
    mine_rate = int(cfg(world.config, "mineRate", 50))
    elapsed = max(0, world.step - int(last_seen))
    return min(cap, energy + mine_rate * elapsed)


def expected_turns_until_row_scrolled(world, row):
    """Conservative scroll-lifetime estimate for a row.

    The true environment tracks a scroll counter. When it is not exposed in
    the observation, starting from the current interval is safer than treating
    `row - southBound` as turns remaining, especially early in the episode.
    """
    episode_steps = int(cfg(world.config, "episodeSteps", 500))
    south = int(world.south)
    step = int(world.step)
    raw_counter = world.obs.get("scrollCounter", None)
    if raw_counter is None:
        inferred_south, inferred_counter = infer_scroll_state_from_step(step, world.config)
        if inferred_south == south:
            counter = inferred_counter
        else:
            counter = 1
    else:
        counter = max(1, int(raw_counter or 1))

    turns = 0
    while south <= row and step + turns < episode_steps:
        turns += 1
        counter -= 1
        if counter <= 0:
            south += 1
            counter = scroll_interval_for_step(step + turns, world.config)
    return turns


def node_lifetime_viable(world, pos, min_turns=MIN_MINE_LIFETIME_TURNS):
    return world.in_bounds(pos) and expected_turns_until_row_scrolled(world, pos[1]) >= min_turns


def move_period_for_robot(robot, config):
    if robot.rtype == SCOUT:
        return int(cfg(config, "scoutMovePeriod", 1))
    if robot.rtype == WORKER:
        return int(cfg(config, "workerMovePeriod", 2))
    if robot.rtype == MINER:
        return int(cfg(config, "minerMovePeriod", 2))
    if robot.rtype == FACTORY:
        return int(cfg(config, "factoryMovePeriod", 2))
    return 2


def estimated_arrival_time(robot, path_dist, config):
    if path_dist is None:
        return None
    if path_dist <= 0:
        return 0
    first_wait = 0 if action_ready(robot.move_cd) else max(1, int(robot.move_cd or 0) - 1)
    return first_wait + 1 + max(0, path_dist - 1) * move_period_for_robot(robot, config)


def unload_cell_orientation_bonus(world, cell):
    if world.factory is None:
        return 0.0
    factory = world.factory
    if cell == (factory.col, factory.row + 1):
        return -140.0
    if cell == (factory.col, factory.row - 1):
        return 20.0
    return 70.0


def factory_liquidity_needed(world):
    factory = world.factory
    if factory is None or not action_ready(factory.build_cd):
        return False
    return bool(pending_build_needs_for_world(world))


def scout_target_count_for_world(world, factory):
    gap = factory.row - world.south
    danger = dynamic_danger_gap_for_step(world.step, world.config)
    if gap < danger + SCOUT_BUILD_MARGIN_SURPLUS:
        return 0
    if world.step < OPENING_SCOUT_STEP_LIMIT:
        target = SCOUT_TARGET_COUNT_OPENING
    elif world.step < 250:
        target = SCOUT_TARGET_COUNT_MID
    elif world.step < 380:
        target = min(SCOUT_TARGET_COUNT_MID, 1)
    else:
        return 0

    return target


def active_scout_count_for_world(world):
    ept = int(cfg(world.config, "energyPerTurn", 1))
    return sum(
        1
        for robot in world.own.values()
        if robot.rtype == SCOUT and robot.energy > ept
    )


def pending_build_needs_for_world(world):
    factory = world.factory
    if factory is None or not action_ready(factory.build_cd):
        return []
    counts = {FACTORY: 0, SCOUT: 0, WORKER: 0, MINER: 0}
    for robot in world.own.values():
        if robot.rtype == SCOUT:
            continue
        counts[robot.rtype] = counts.get(robot.rtype, 0) + 1
    counts[SCOUT] = active_scout_count_for_world(world)

    needs = []
    if counts.get(WORKER, 0) < WORKER_TARGET_COUNT:
        needs.append(int(cfg(world.config, "workerCost", 200)) + FACTORY_WORKER_RESERVE)

    factory_gap = factory.row - world.south
    danger = dynamic_danger_gap_for_step(world.step, world.config)
    has_node = any(
        node_lifetime_viable(world, pos)
        and pos not in world.state["known_mines"]
        for pos in world.state["known_nodes"]
    )
    if (
        has_node
        and counts.get(MINER, 0) < MINER_TARGET_COUNT
        and factory_gap >= danger + MINER_BUILD_MARGIN
    ):
        needs.append(int(cfg(world.config, "minerCost", 300)) + FACTORY_MINER_RESERVE)

    if counts.get(SCOUT, 0) < scout_target_count_for_world(world, factory):
        needs.append(int(cfg(world.config, "scoutCost", 50)) + FACTORY_SCOUT_RESERVE)

    return [need for need in needs if factory.energy < need]


def transfer_would_cross_build_threshold(world, robot):
    factory = world.factory
    if factory is None:
        return False
    ept = int(cfg(world.config, "energyPerTurn", 1))
    # Transfer is processed after the factory build phase, so a transfer cannot
    # unlock a build on the current turn. This estimate asks whether the energy
    # would cross a pending build threshold on the next build opportunity after
    # both source and factory pay upkeep.
    energy_for_next_build = factory.energy - ept + max(0, robot.energy - ept) - ept
    return any(factory.energy < need <= energy_for_next_build for need in pending_build_needs_for_world(world))


def scout_has_good_forward_job(world, robot):
    if robot.rtype != SCOUT:
        return False
    cap = MAX_ENERGY_BY_TYPE.get(SCOUT, 100)
    if robot.energy >= SCOUT_RETURN_ENERGY:
        # Simple-ferry profile: once a scout carries useful bankable energy,
        # generic frontier exploration is no longer a good forward job. Only
        # a very nearby, high-value resource can justify one more step out.
        for cpos, energy in world.visible_crystals.items():
            if cpos[1] >= robot.row and energy >= 45 and manhattan(robot.pos, cpos) <= 1:
                return True
        return False
    if robot.energy >= cap - 5:
        # A capped scout should bank energy instead of treating generic
        # frontier exploration as productive work. Only a very nearby,
        # high-value resource can justify staying forward at cap.
        for cpos, energy in world.visible_crystals.items():
            if cpos[1] >= robot.row and energy >= 45 and manhattan(robot.pos, cpos) <= 2:
                return True
        for mine_pos, mine_data in world.state["known_mines"].items():
            if not world.in_bounds(mine_pos):
                continue
            energy = estimated_mine_energy(world, mine_pos, mine_data)
            _, _, owner, _ = parse_mine_record(mine_data)
            try:
                owner = int(owner)
            except Exception:
                continue
            if (
                owner == world.player
                and energy >= max(250, MINE_HARVEST_MIN_ENERGY)
                and manhattan(robot.pos, mine_pos) <= 2
            ):
                return True
        return False
    if robot.energy < cap - 5:
        for cpos, energy in world.visible_crystals.items():
            if cpos[1] >= robot.row and energy >= 20 and manhattan(robot.pos, cpos) <= 6:
                return True
        for mine_pos, mine_data in world.state["known_mines"].items():
            if not world.in_bounds(mine_pos):
                continue
            energy = estimated_mine_energy(world, mine_pos, mine_data)
            _, _, owner, _ = parse_mine_record(mine_data)
            try:
                owner = int(owner)
            except Exception:
                continue
            if owner == world.player and energy >= MINE_HARVEST_MIN_ENERGY and manhattan(robot.pos, mine_pos) <= 8:
                return True
    for fpos in world.known_frontiers():
        if fpos[1] >= robot.row and manhattan(robot.pos, fpos) <= 7:
            return True
    return False


def should_seek_scout_unload(world, robot):
    if robot.rtype != SCOUT:
        return False
    danger = dynamic_danger_gap_for_step(world.step, world.config)
    if robot.row - world.south <= danger + 1:
        return True
    if robot.energy >= SCOUT_FORCE_RETURN_ENERGY:
        return True
    if robot.energy >= SCOUT_RETURN_ENERGY and not scout_has_good_forward_job(world, robot):
        return True
    if not wants_to_unload(robot):
        return False
    if transfer_would_cross_build_threshold(world, robot):
        return True
    cap = MAX_ENERGY_BY_TYPE.get(SCOUT, 100)
    return robot.energy >= cap - 2 and not scout_has_good_forward_job(world, robot)
# ============================================================
# Mechanics / Physics
# ============================================================


def step_pos(pos, direction, distance=1):
    dc, dr, _ = DIRS[direction]
    return pos[0] + dc * distance, pos[1] + dr * distance


def mirror_pos(pos, width):
    return (width - 1 - pos[0], pos[1])


def mirror_wall_bits(wall):
    return (wall & WALL_N) | (wall & WALL_S) | ((wall & WALL_E) << 2) | ((wall & WALL_W) >> 2)


def is_fixed_wall(pos, direction, width):
    col = pos[0]
    half = width // 2
    return (
        (direction == "WEST" and col == 0)
        or (direction == "EAST" and col == width - 1)
        or (direction == "EAST" and col == half - 1)
        or (direction == "WEST" and col == half)
    )


def known_edge_blocked(world, pos, direction):
    """Return True if either known side of an edge has the wall bit set."""
    if direction not in DIRS:
        return False
    _, _, bit = DIRS[direction]
    wall = world.wall_at(pos)
    if wall is not None and (wall & bit):
        return True
    nxt = step_pos(pos, direction)
    if not world.in_bounds(nxt):
        return False
    opposite_wall = world.wall_at(nxt)
    if opposite_wall is None:
        return False
    opposite_bit = DIRS[OPPOSITE[direction]][2]
    return bool(opposite_wall & opposite_bit)


def crush_danger(robot, enemy):
    """True when ending on the enemy cell is likely bad for this robot."""
    if enemy is None:
        return False
    if robot.rtype == FACTORY and enemy.rtype != FACTORY:
        return False
    return COMBAT_RANK.get(enemy.rtype, 0) >= COMBAT_RANK.get(robot.rtype, 0)
# ============================================================
# World Model
# ============================================================


def fresh_state(player):
    return {
        "player": player,
        "step": -1,
        "southBound": 0,
        "known_walls": {},
        "known_nodes": {},
        "known_mines": {},
        "seen_enemies": {},
        "robot_history": {},
        "assigned_targets": {},
        "last_actions": {},
    }


class WorldModel:
    """Observation normalizer and durable memory.

    This class answers "what do we know?" rather than "what should we do?".
    Strategy choices such as sacrificing a friendly blocker stay in Strategy.
    That boundary keeps wall indexing, mine memory, symmetry, and fog-of-war
    persistence reusable across strategy experiments.
    """

    def __init__(self, obs, config, state_store=None):
        self.obs = to_plain_dict(obs)
        self.config = to_plain_dict(config)
        self.state_store = STATE_BY_PLAYER if state_store is None else state_store
        self.player = int(self.obs.get("player", 0))
        self.width = int(cfg(self.config, "width", 20))
        self.height = int(cfg(self.config, "height", 20))
        self.step = self._infer_step()
        self.south = self._infer_south_bound()
        self.north = self._infer_north_bound()
        self.state = self._state_for_player()

        self.own = {}
        self.enemies = {}
        self.own_positions = {}
        self.enemy_positions = {}
        self.visible_crystals = {}
        self.visible_nodes = set()
        self.visible_cells = set()
        self.factory = None

        self.update_memory()

    def _infer_step(self):
        raw_step = self.obs.get("step", None)
        if raw_step is not None:
            return int(raw_step or 0)
        if self._looks_like_initial_observation():
            return 0
        previous = self.state_store.get(self.player, {}).get("step", -1)
        return previous + 1

    def _infer_south_bound(self):
        raw_south = self.obs.get("southBound", None)
        if raw_south is not None:
            return int(raw_south or 0)
        return infer_south_bound_from_step(self.step, self.config)

    def _infer_north_bound(self):
        raw_north = self.obs.get("northBound", None)
        if raw_north is not None:
            return int(raw_north or 0)
        return self.south + self.height - 1

    def _looks_like_initial_observation(self):
        robots = self.obs.get("robots") or {}
        factory_energy = int(cfg(self.config, "factoryEnergy", 1000))
        owned = []
        for uid, data in robots.items():
            try:
                robot = robot_record(uid, data)
            except Exception:
                continue
            if robot.owner == self.player:
                owned.append(robot)
        if len(owned) != 1:
            return False
        robot = owned[0]
        raw_south = self.obs.get("southBound", None)
        south = int(raw_south or 0)
        if raw_south is not None and south != 0:
            return False
        return (
            robot.rtype == FACTORY
            and robot.row - south <= 2
            and robot.energy >= int(factory_energy * 0.90)
        )

    def _state_for_player(self):
        state = self.state_store.get(self.player)
        should_reset = (
            state is None
            or self.step < state.get("step", -1)
            or self.south < state.get("southBound", 0)
        )
        if should_reset:
            state = fresh_state(self.player)
            self.state_store[self.player] = state
        state["step"] = self.step
        state["southBound"] = self.south
        return state

    def update_memory(self):
        self._read_robots()
        self._read_visible_cells()
        self._read_walls()
        self._read_resources()
        self._prune_memory()

    def _read_robots(self):
        robots = self.obs.get("robots") or {}
        for uid, data in robots.items():
            try:
                robot = robot_record(uid, data)
            except Exception:
                continue
            if robot.owner == self.player:
                self.own[robot.uid] = robot
                self.own_positions[robot.pos] = robot.uid
                if robot.rtype == FACTORY:
                    self.factory = robot
                history = self.state["robot_history"].setdefault(robot.uid, [])
                history.append(robot.pos)
                if len(history) > 12:
                    del history[:-12]
            else:
                self.enemies[robot.uid] = robot
                self.enemy_positions[robot.pos] = robot.uid
                self.state["seen_enemies"][robot.uid] = (self.step, robot)

        for uid in list(self.state["assigned_targets"]):
            if uid not in self.own:
                del self.state["assigned_targets"][uid]
        for uid in list(self.state["robot_history"]):
            if uid not in self.own:
                del self.state["robot_history"][uid]

    def _read_visible_cells(self):
        for robot in self.own.values():
            radius = int(cfg(self.config, "vision" + TYPE_NAMES.get(robot.rtype, "").title(), 0))
            if radius <= 0:
                radius = VISION_BY_TYPE.get(robot.rtype, 3)
            for dc in range(-radius, radius + 1):
                remaining = radius - abs(dc)
                for dr in range(-remaining, remaining + 1):
                    pos = (robot.col + dc, robot.row + dr)
                    if self.in_bounds(pos):
                        self.visible_cells.add(pos)

    def _read_walls(self):
        walls = self.obs.get("walls") or []
        observed = {}
        for idx, wall in enumerate(walls):
            try:
                wall = int(wall)
            except Exception:
                continue
            if wall < 0:
                continue
            pos = (idx % self.width, self.south + idx // self.width)
            if self.in_bounds(pos):
                observed[pos] = wall
                self.state["known_walls"][pos] = wall
        # A wall is an edge fact. If one side is observed and the opposite
        # cell is remembered, synchronize the reciprocal bit so BFS does not
        # disagree depending on which endpoint it expands from.
        for pos, wall in observed.items():
            for direction, (_, _, bit) in DIRS.items():
                nxt = step_pos(pos, direction)
                if (
                    not self.in_bounds(nxt)
                    or nxt in observed
                    or nxt not in self.state["known_walls"]
                ):
                    continue
                opposite_bit = DIRS[OPPOSITE[direction]][2]
                neighbor_wall = int(self.state["known_walls"][nxt])
                if wall & bit:
                    neighbor_wall |= opposite_bit
                else:
                    neighbor_wall &= ~opposite_bit
                self.state["known_walls"][nxt] = neighbor_wall
        # The maze is east/west symmetric. Mirrored walls are inserted only
        # as defaults; a later direct observation always wins.
        for pos, wall in observed.items():
            mirror = mirror_pos(pos, self.width)
            if not self.in_bounds(mirror) or mirror in observed:
                continue
            self.state["known_walls"].setdefault(mirror, mirror_wall_bits(wall))

    def _read_resources(self):
        self.visible_crystals = {}
        for key, value in (self.obs.get("crystals") or {}).items():
            pos = parse_pos_key(key)
            if pos is not None and self.in_bounds(pos):
                self.visible_crystals[pos] = int(value)

        self.visible_nodes = set()
        for key in (self.obs.get("miningNodes") or {}):
            pos = parse_pos_key(key)
            if pos is not None and self.in_bounds(pos):
                self.visible_nodes.add(pos)
                self.state["known_nodes"][pos] = self.step

        seen_mine_positions = set()
        for key, value in (self.obs.get("mines") or {}).items():
            pos = parse_pos_key(key)
            if pos is None or not self.in_bounds(pos):
                continue
            seen_mine_positions.add(pos)
            if not isinstance(value, (list, tuple)):
                self.state["known_mines"][pos] = value
                continue

            values = list(value) + [0, 0, None]
            energy = int(values[0])
            max_energy = int(values[1])
            owner = values[2]
            old = self.state["known_mines"].get(pos)
            if pos in self.visible_cells or old is None:
                self.state["known_mines"][pos] = [energy, max_energy, owner, self.step]
            else:
                # `obs.mines` can contain remembered last-known records. Do
                # not refresh last_seen unless the mine is in current vision;
                # otherwise hidden friendly mine energy is biased downward.
                old_energy, old_max, old_owner, old_seen = parse_mine_record(old)
                self.state["known_mines"][pos] = [
                    old_energy,
                    max_energy if max_energy > 0 else old_max,
                    owner if owner is not None else old_owner,
                    old_seen,
                ]

        for pos in seen_mine_positions:
            self.state["known_nodes"].pop(pos, None)

        # If a remembered node is currently visible but no node or mine remains,
        # treat it as consumed.
        for pos in list(self.state["known_nodes"]):
            if (
                pos in self.visible_cells
                and pos not in self.visible_nodes
                and pos not in seen_mine_positions
            ):
                del self.state["known_nodes"][pos]

    def _prune_memory(self):
        prune_before = self.south - 2
        for table_name in ("known_walls", "known_nodes", "known_mines"):
            table = self.state[table_name]
            for pos in list(table):
                if pos[1] < prune_before:
                    del table[pos]
        for uid in list(self.state["seen_enemies"]):
            turn, robot = self.state["seen_enemies"][uid]
            if self.step - turn > 20 or robot.row < prune_before:
                del self.state["seen_enemies"][uid]

    def in_bounds(self, pos):
        return 0 <= pos[0] < self.width and self.south <= pos[1] <= self.north

    def wall_at(self, pos):
        return self.state["known_walls"].get(pos)

    def current_wall(self, robot):
        return self.wall_at(robot.pos)

    def can_step(self, pos, direction, allow_unknown_target=False):
        if not self.in_bounds(pos):
            return False
        wall = self.wall_at(pos)
        if wall is None:
            return False
        if known_edge_blocked(self, pos, direction):
            return False
        nxt = step_pos(pos, direction)
        if not self.in_bounds(nxt):
            return False
        if allow_unknown_target:
            return True
        return self.wall_at(nxt) is not None

    def neighbors(self, pos, allow_unknown_target=False):
        for direction in DIR_ORDER:
            if self.can_step(pos, direction, allow_unknown_target=allow_unknown_target):
                yield direction, step_pos(pos, direction)

    def enemy_at(self, pos):
        uid = self.enemy_positions.get(pos)
        return self.enemies.get(uid) if uid is not None else None

    def own_at(self, pos):
        uid = self.own_positions.get(pos)
        return self.own.get(uid) if uid is not None else None

    def safe_destination(self, robot, pos, reserved_next):
        if pos in reserved_next:
            return False
        occupant = self.own_at(pos)
        if occupant is not None and occupant.uid != robot.uid:
            return False
        enemy = self.enemy_at(pos)
        if crush_danger(robot, enemy):
            return False
        return True

    def unload_cells_for_factory(self):
        if self.factory is None:
            return []
        cells = []
        for direction in DIR_ORDER:
            cell = step_pos(self.factory.pos, direction)
            if not self.in_bounds(cell):
                continue
            if self.wall_at(cell) is None:
                continue
            if self.can_step(cell, OPPOSITE[direction], allow_unknown_target=True):
                cells.append(cell)
        return cells

    def known_frontiers(self):
        frontiers = set()
        for pos, wall in self.state["known_walls"].items():
            if not self.in_bounds(pos):
                continue
            for direction, (dc, dr, bit) in DIRS.items():
                if wall & bit:
                    continue
                nxt = (pos[0] + dc, pos[1] + dr)
                if self.in_bounds(nxt) and nxt not in self.state["known_walls"]:
                    frontiers.add(pos)
                    break
        return frontiers

    def frontier_exit(self, robot, reserved_next):
        wall = self.wall_at(robot.pos)
        if wall is None:
            return None, None
        options = []
        for direction, (dc, dr, bit) in DIRS.items():
            if wall & bit:
                continue
            nxt = (robot.col + dc, robot.row + dr)
            if not self.in_bounds(nxt):
                continue
            if nxt in self.state["known_walls"]:
                continue
            if not self.safe_destination(robot, nxt, reserved_next):
                continue
            score = (nxt[1] - robot.row, -abs(nxt[0] - self.width // 2), direction == "NORTH")
            options.append((score, direction, nxt))
        options.sort(reverse=True)
        if options:
            _, direction, nxt = options[0]
            return direction, nxt
        return None, None

    def safe_moves(self, robot, reserved_next, prefer_north=True):
        options = []
        for direction, nxt in self.neighbors(robot.pos, allow_unknown_target=True):
            if not self.safe_destination(robot, nxt, reserved_next):
                continue
            progress = nxt[1] - robot.row
            center_pull = -abs(nxt[0] - self.width // 2)
            if prefer_north:
                score = (progress, center_pull, direction == "NORTH")
            else:
                score = (center_pull, progress, direction == "NORTH")
            options.append((score, direction, nxt))
        options.sort(reverse=True)
        return [(direction, nxt) for _, direction, nxt in options]
# ============================================================
# Pathing
# ============================================================


def bfs_first_step_and_distance(
    world,
    robot,
    goals,
    reserved_next,
    max_nodes=MAX_BFS_NODES,
    allow_occupied_goal=False,
):
    goals = set(goals)
    if not goals:
        return None, None, None
    start = robot.pos
    if start in goals:
        return "IDLE", start, 0

    queue = deque([start])
    first_dir = {start: None}
    distance = {start: 0}
    seen = {start}
    nodes = 0

    while queue and nodes < max_nodes:
        cur = queue.popleft()
        nodes += 1
        for direction, nxt in world.neighbors(cur, allow_unknown_target=False):
            if nxt in seen:
                continue
            if nxt != start and nxt in reserved_next:
                continue
            # Most pathing should avoid currently occupied friendly cells.
            # The exception is ROI/pathing toward a miner that is about to
            # TRANSFORM: the miner disappears before movement, so its current
            # cell can be a valid same-turn mine-harvest destination.
            if allow_occupied_goal and nxt in goals:
                if crush_danger(robot, world.enemy_at(nxt)):
                    continue
                return direction if cur == start else first_dir[cur], nxt, distance[cur] + 1
            if nxt != start and nxt in world.own_positions:
                continue
            if crush_danger(robot, world.enemy_at(nxt)):
                continue
            seen.add(nxt)
            first_dir[nxt] = direction if cur == start else first_dir[cur]
            distance[nxt] = distance[cur] + 1
            if nxt in goals:
                return first_dir[nxt], nxt, distance[nxt]
            queue.append(nxt)
    return None, None, None


def bfs_first_step(
    world,
    robot,
    goals,
    reserved_next,
    max_nodes=MAX_BFS_NODES,
    allow_occupied_goal=False,
):
    direction, pos, _ = bfs_first_step_and_distance(
        world,
        robot,
        goals,
        reserved_next,
        max_nodes=max_nodes,
        allow_occupied_goal=allow_occupied_goal,
    )
    return direction, pos


def factory_optimistic_can_step(world, pos, direction):
    """Factory-only planning edge test that treats unknown future cells as open.

    The first committed step is still checked by the normal safety layer. This
    helper only helps the factory choose a purposeful long-horizon side step
    when fog hides the rest of the corridor. Known wall bits and fixed border
    walls remain hard blockers.
    """
    if direction not in DIRS or not world.in_bounds(pos):
        return False
    if is_fixed_wall(pos, direction, world.width):
        return False
    if known_edge_blocked(world, pos, direction):
        return False
    nxt = step_pos(pos, direction)
    return world.in_bounds(nxt)


def factory_optimistic_bfs_first_step(
    world,
    robot,
    start,
    goals,
    reserved_next,
    max_nodes=FACTORY_OPTIMISTIC_BFS_NODES,
    allow_friendly_future=False,
):
    """Long-horizon north search for the factory under fog of war.

    Unknown future cells are treated as open for route scoring. The selected
    first step is still checked by the normal safety layer before execution.
    """
    goals = set(goals)
    if not goals or not world.in_bounds(start):
        return None, None
    if start in goals:
        return "IDLE", start

    queue = deque([start])
    first_dir = {start: None}
    seen = {start}
    nodes = 0

    while queue and nodes < max_nodes:
        cur = queue.popleft()
        nodes += 1
        for direction in DIR_ORDER:
            if not factory_optimistic_can_step(world, cur, direction):
                continue
            nxt = step_pos(cur, direction)
            if nxt in seen:
                continue
            if nxt != start and nxt in reserved_next:
                continue
            if (
                not allow_friendly_future
                and nxt != start
                and nxt in world.own_positions
            ):
                continue
            if crush_danger(robot, world.enemy_at(nxt)):
                continue
            seen.add(nxt)
            first_dir[nxt] = direction if cur == start else first_dir[cur]
            if nxt in goals:
                return first_dir[nxt], nxt
            queue.append(nxt)
    return None, None


def factory_optimistic_jump_bfs_first_action(
    world,
    robot,
    start,
    goals,
    reserved_next,
    jump_cd,
    max_nodes=FACTORY_JUMP_AWARE_BFS_NODES,
    allow_friendly_future=False,
):
    """Factory route search that treats jump as a future path option.

    The returned action is only the first primitive action. A future jump can
    make an ordinary first step valuable, while an immediate jump is returned
    only when the current cooldown state allows it.
    """
    goals = set(goals)
    if not goals or not world.in_bounds(start):
        return None, None
    if start in goals:
        return "IDLE", start

    initial_jump_cd = max(0, int(jump_cd or 0))
    queue = deque([(start, initial_jump_cd, None, None, 0)])
    seen = {(start, initial_jump_cd)}
    nodes = 0

    while queue and nodes < max_nodes:
        pos, current_jump_cd, first_action, first_pos, depth = queue.popleft()
        nodes += 1
        if pos in goals and depth > 0:
            return first_action, first_pos
        if depth >= FACTORY_BFS_LOOKAHEAD:
            continue

        for direction in DIR_ORDER:
            if direction == "SOUTH":
                continue
            if not factory_optimistic_can_step(world, pos, direction):
                continue
            nxt = step_pos(pos, direction)
            if nxt != start and nxt in reserved_next:
                continue
            if (
                not allow_friendly_future
                and nxt != start
                and nxt in world.own_positions
            ):
                continue
            if crush_danger(robot, world.enemy_at(nxt)):
                continue
            next_jump_cd = max(0, current_jump_cd - 1)
            key = (nxt, next_jump_cd)
            if key in seen:
                continue
            seen.add(key)
            queue.append(
                (
                    nxt,
                    next_jump_cd,
                    first_action or direction,
                    first_pos or nxt,
                    depth + 1,
                )
            )

        if current_jump_cd <= 1:
            for direction in ("NORTH", "EAST", "WEST"):
                if depth == 0 and REQUIRE_MOVE_READY_FOR_JUMP and not action_ready(robot.move_cd):
                    continue
                landing = step_pos(pos, direction, distance=2)
                if not world.in_bounds(landing):
                    continue
                if landing != start and landing in reserved_next:
                    continue
                if (
                    not allow_friendly_future
                    and landing != start
                    and landing in world.own_positions
                ):
                    continue
                if crush_danger(robot, world.enemy_at(landing)):
                    continue
                wall = world.wall_at(landing)
                if wall is not None and not any(
                    world.can_step(landing, d, allow_unknown_target=True)
                    for d in DIR_ORDER
                ):
                    continue
                key = (landing, 20)
                if key in seen:
                    continue
                seen.add(key)
                jump_action = "JUMP_" + direction
                queue.append(
                    (
                        landing,
                        20,
                        first_action or jump_action,
                        first_pos or landing,
                        depth + 1,
                    )
                )
    return None, None


def worker_has_immediate_wall_job(world, robot, allow_default=False):
    if robot.rtype != WORKER:
        return False
    wall = world.wall_at(robot.pos)
    if wall is None:
        return False
    cost = int(cfg(world.config, "wallRemoveCost", 100))
    if robot.energy < cost + WALL_REMOVE_BUFFER:
        return False

    def can_remove(direction):
        nxt = step_pos(robot.pos, direction)
        return (
            known_edge_blocked(world, robot.pos, direction)
            and world.in_bounds(nxt)
            and not is_fixed_wall(robot.pos, direction, world.width)
        )

    factory = world.factory
    if factory is not None and manhattan(robot.pos, factory.pos) == 1:
        for direction in DIR_ORDER:
            if step_pos(robot.pos, direction) == factory.pos and can_remove(direction):
                return True

    if factory is not None:
        corridor = {
            factory.pos,
            (factory.col, factory.row + 1),
            (factory.col - 1, factory.row + 1),
            (factory.col + 1, factory.row + 1),
        }
        if robot.pos in corridor and can_remove("NORTH"):
            return True
        for direction in DIR_ORDER:
            if step_pos(robot.pos, direction) in corridor and can_remove(direction):
                return True

    return allow_default and can_remove("NORTH")


def miner_has_known_economy_job(world, robot, danger):
    if robot.rtype != MINER:
        return False
    if (
        robot.pos in world.visible_nodes
        and robot.pos not in world.state["known_mines"]
        and node_lifetime_viable(world, robot.pos)
    ):
        return True

    for node in world.state["known_nodes"]:
        if node in world.state["known_mines"]:
            continue
        if node_lifetime_viable(world, node):
            return True

    for mine_pos, mine_data in world.state["known_mines"].items():
        if not world.in_bounds(mine_pos):
            continue
        energy = estimated_mine_energy(world, mine_pos, mine_data)
        _, _, owner, _ = parse_mine_record(mine_data)
        try:
            owner = int(owner)
        except Exception:
            continue
        if (
            owner == world.player
            and energy >= MINE_HARVEST_MIN_ENERGY
            and node_lifetime_viable(world, mine_pos, min_turns=max(MINE_HARVEST_SCROLL_GAP, 4))
        ):
            return True
    return False


def select_candidate_targets(world, robot, reserved_targets):
    """Enumerate transparent, scoreable intents for one robot.

    Scores are heuristic and intentionally inspectable. They give paired-seed
    tuning a clear surface while final legality remains enforced by
    Strategy.commit().
    """

    targets = []
    pos = robot.pos
    danger_gap = robot.row - world.south
    factory_gap = (
        world.factory.row - world.south
        if world.factory is not None
        else 999
    )
    danger = dynamic_danger_gap_for_step(world.step, world.config)
    caution = danger + 3

    # Scroll escape is a hard-priority intent. If a unit is already inside the
    # danger band, returning only an escape target prevents a crystal or mine
    # score from overriding basic survival.
    if danger_gap <= danger:
        northish = [
            p
            for p in world.state["known_walls"]
            if p[1] >= robot.row + 2 and world.in_bounds(p)
        ]
        if northish:
            best = min(northish, key=lambda p: (manhattan(pos, p), -p[1]))
            return [Target("scroll_escape", best, 10_000.0)]

    if should_seek_scout_unload(world, robot):
        for cell in world.unload_cells_for_factory():
            if cell in reserved_targets:
                continue
            dist = max(1, manhattan(pos, cell))
            score = 1200.0 - 6.0 * dist + unload_cell_orientation_bonus(world, cell)
            targets.append(Target("unload_factory", cell, score))

    # Worker/miner unload is intentionally narrower than scout unload. These
    # units carry job capital: a 0-energy worker cannot open a wall, and a
    # 0-energy miner cannot transform. Transfer is useful mainly when it
    # crosses a factory build threshold or prevents scroll loss.
    if (
        robot.rtype in (WORKER, MINER)
        and wants_to_unload(robot)
        and world.factory is not None
        and transfer_would_cross_build_threshold(world, robot)
        and robot.pos != step_pos(world.factory.pos, "NORTH")
        and not worker_has_immediate_wall_job(world, robot)
        and not miner_has_known_economy_job(world, robot, danger)
    ):
        for cell in world.unload_cells_for_factory():
            if cell in reserved_targets:
                continue
            dist = max(1, manhattan(pos, cell))
            score = 520.0 - 6.5 * dist + unload_cell_orientation_bonus(world, cell) * 0.35
            targets.append(Target("unload_factory", cell, score))

    if robot.rtype in (SCOUT, WORKER, MINER):
        cap = MAX_ENERGY_BY_TYPE.get(robot.rtype)
        if cap and robot.energy < cap * UNLOAD_FRACTION:
            for mine_pos, mine_data in world.state["known_mines"].items():
                if mine_pos in reserved_targets or not world.in_bounds(mine_pos):
                    continue
                energy = estimated_mine_energy(world, mine_pos, mine_data)
                _, _, owner, _ = parse_mine_record(mine_data)
                try:
                    owner = int(owner)
                except Exception:
                    continue
                if owner != world.player:
                    continue
                if energy < MINE_HARVEST_MIN_ENERGY:
                    continue
                if not node_lifetime_viable(world, mine_pos, min_turns=max(MINE_HARVEST_SCROLL_GAP, 4)):
                    continue
                dist = max(1, manhattan(pos, mine_pos))
                score = 650.0 + 0.4 * energy - 9.0 * dist + 0.15 * (mine_pos[1] - pos[1])
                targets.append(Target("mine_harvest", mine_pos, score))

    if robot.rtype == MINER:
        for node in world.state["known_nodes"]:
            if node in reserved_targets or not world.in_bounds(node):
                continue
            if node in world.state["known_mines"]:
                continue
            if not node_lifetime_viable(world, node):
                continue
            lifetime = expected_turns_until_row_scrolled(world, node[1])
            score = 800.0 + 2.0 * lifetime - manhattan(pos, node) + NODE_NORTH_BONUS * (node[1] - pos[1])
            targets.append(Target("node", node, score))

    if robot.rtype == WORKER and world.factory is not None:
        f = world.factory
        corridor = [
            f.pos,
            (f.col, f.row + 1),
            (f.col - 1, f.row + 1),
            (f.col + 1, f.row + 1),
        ]
        corridor_jobs = []
        seen_corridor_jobs = set()

        def add_corridor_job(actor_pos, priority):
            if actor_pos in seen_corridor_jobs:
                return
            seen_corridor_jobs.add(actor_pos)
            corridor_jobs.append((actor_pos, priority))

        for cpos in corridor:
            if not world.in_bounds(cpos):
                continue
            wall = world.wall_at(cpos)
            if wall is None:
                continue
            if wall & WALL_N:
                add_corridor_job(cpos, 1000.0)
                opposite = step_pos(cpos, "NORTH")
                opposite_wall = world.wall_at(opposite)
                if (
                    world.in_bounds(opposite)
                    and opposite_wall is not None
                    and (opposite_wall & WALL_S)
                ):
                    add_corridor_job(opposite, 980.0)
            if step_pos(cpos, "SOUTH") == f.pos and (wall & WALL_S):
                add_corridor_job(cpos, 980.0)

        for cpos, base_score in corridor_jobs:
            if cpos in reserved_targets or not world.in_bounds(cpos):
                continue
            dist = max(1, manhattan(pos, cpos))
            score = base_score - 10.0 * dist + 0.5 * (cpos[1] - pos[1])
            targets.append(Target("factory_corridor_open", cpos, score))

    if robot.rtype == WORKER and world.factory is not None:
        f = world.factory
        if factory_gap <= danger + WORKER_VANGUARD_GAP:
            vanguard = (f.col, min(world.north, f.row + WORKER_VANGUARD_AHEAD))
            if vanguard not in reserved_targets and world.in_bounds(vanguard):
                dist = max(1, manhattan(pos, vanguard))
                score = WORKER_VANGUARD_SCORE - 4.0 * dist + 0.4 * (vanguard[1] - pos[1])
                targets.append(Target("worker_vanguard", vanguard, score))

    if robot.rtype in (SCOUT, WORKER, MINER):
        if robot.rtype == WORKER:
            allow_crystal = (
                robot.energy < WORKER_REFILL_ENERGY
                and factory_gap >= danger + WORKER_REFILL_SAFE_GAP
            )
        else:
            allow_crystal = True
        for cpos, energy in sorted(
            world.visible_crystals.items(),
            key=lambda item: -item[1],
        )[:MAX_TARGETS_PER_KIND]:
            if not allow_crystal:
                continue
            if cpos in reserved_targets or not world.in_bounds(cpos):
                continue
            if robot.rtype in MAX_ENERGY_BY_TYPE and robot.energy >= MAX_ENERGY_BY_TYPE[robot.rtype] - 3:
                continue
            dist = max(1, manhattan(pos, cpos))
            worker_refill = (
                robot.rtype == WORKER
                and robot.energy < WORKER_REFILL_ENERGY
                and factory_gap >= danger + WORKER_REFILL_SAFE_GAP
            )
            if robot.rtype == WORKER and robot.energy < WORKER_REFILL_ENERGY and dist > WORKER_REFILL_MAX_CRYSTAL_DIST:
                continue
            if robot.rtype == SCOUT:
                base = 560.0
            elif robot.rtype == WORKER:
                base = 320.0
            else:
                base = 250.0
            score = base + CRYSTAL_VALUE_WEIGHT * energy - CRYSTAL_DIST_PENALTY * dist + 0.2 * (cpos[1] - pos[1])
            if worker_refill:
                score += WORKER_REFILL_CRYSTAL_BONUS
            targets.append(Target("crystal", cpos, score))

    frontiers = sorted(
        world.known_frontiers(),
        key=lambda p: (
            -p[1],
            manhattan(pos, p),
            abs(p[0] - world.width // 2),
            p[0],
        ),
    )
    for fpos in frontiers[:MAX_TARGETS_PER_KIND * 3]:
        if fpos in reserved_targets or not world.in_bounds(fpos):
            continue
        dist = max(1, manhattan(pos, fpos))
        north_gain = fpos[1] - pos[1]
        score = 300.0 - 5.5 * dist + FRONTIER_NORTH_BONUS * north_gain
        if robot.rtype == SCOUT:
            score += 60.0
        elif robot.rtype == WORKER:
            score += 30.0
        targets.append(Target("frontier", fpos, score))

    if robot.rtype == WORKER:
        wall = world.wall_at(pos)
        if (
            wall is not None
            and (wall & WALL_N)
            and (danger_gap <= danger or factory_gap > caution + 2)
        ):
            north = (pos[0], pos[1] + 1)
            if world.in_bounds(north):
                targets.append(Target("wall_open", pos, 260.0))

    if targets:
        targets.sort(key=lambda item: item.score, reverse=True)
        return targets

    fallback_row = min(world.north, robot.row + 4)
    return [Target("north_drift", (robot.col, fallback_row), 1.0)]


def select_best_target(world, robot, reserved_targets):
    return select_candidate_targets(world, robot, reserved_targets)[0]
# ============================================================
# Strategy
# ============================================================


class Strategy:
    """Heuristic policy plus deterministic safety shield.

    Policy methods choose useful intents under current heuristic preferences.
    Commit and normalize methods enforce the rules and prevent friendly
    collisions. Replacing the first half is safe; replacing the second half with
    unchecked primitive actions is not.
    """

    def __init__(self, world, started, deadline):
        self.world = world
        self.started = started
        self.deadline = deadline
        self.actions = {}
        # Positions already claimed for the end of this turn. This is the main
        # friendly-fire guard: a policy may want a cell, but commit decides
        # whether another robot has already made that cell unsafe.
        self.reserved_next = set()
        # Target reservation is softer than movement reservation. It reduces
        # duplicate long-range assignments, while still allowing the final
        # movement shield to make the exact collision decision.
        self.reserved_targets = set()
        # Forced actions are tactical side effects created by factory movement:
        # a blocker may be asked to vacate, or a miner may be forced to
        # transform so a carrier can enter the newly created mine cell.
        self.forced_actions = {}
        self.forced_reserved_next = {}
        self.sacrifice_uids = set()
        self.counts = self._unit_counts()
        self.planned_special_actions = {}
        self.emergency_jump_uids = set()

    def expired(self):
        return time.perf_counter() > self.deadline - MIN_TIME_REMAINING

    def _unit_counts(self):
        counts = {FACTORY: 0, SCOUT: 0, WORKER: 0, MINER: 0}
        ept = self.energy_per_turn()
        for robot in self.world.own.values():
            if robot.rtype == SCOUT and robot.energy <= ept:
                continue
            counts[robot.rtype] = counts.get(robot.rtype, 0) + 1
        return counts

    def energy_per_turn(self):
        return int(cfg(self.world.config, "energyPerTurn", 1))

    def has_energy_after_upkeep(self, robot):
        return robot.energy > self.energy_per_turn()

    def can_pay_after_upkeep(self, robot, cost):
        return robot.energy >= int(cost) + self.energy_per_turn()

    def plan(self):
        danger = self.dynamic_danger_gap()
        # Decide special-vacating actions before movement safety checks. This
        # keeps the safety layer dependent on an explicit plan instead of
        # calling transform policy while evaluating occupied cells.
        self.planned_special_actions = self.preplan_special_actions()

        # The factory and units near the scroll line are planned first because
        # their reservations define the constraints for less urgent robots.
        robots = sorted(
            self.world.own.values(),
            key=lambda robot: (
                0 if robot.rtype == FACTORY else 1 if robot.row - self.world.south <= danger else 2,
                0 if robot.rtype == WORKER else 1 if robot.rtype == MINER else 2 if robot.rtype == SCOUT else 3,
                robot.row,
                robot.uid,
            ),
        )

        for robot in robots:
            if robot.uid in self.forced_actions:
                action, next_pos = self.forced_actions[robot.uid]
                self.commit(robot, action, next_pos)
                continue
            if robot.uid in self.planned_special_actions:
                action, next_pos = self.planned_special_actions[robot.uid]
                self.commit(robot, action, next_pos)
                continue
            if self.expired():
                self.commit(robot, "IDLE", robot.pos)
                continue
            action, next_pos = self.policy_action(robot)
            self.commit(robot, action, next_pos)

        self.world.state["last_actions"] = dict(self.actions)
        return self.actions

    def policy_action(self, robot):
        if robot.rtype == FACTORY:
            return self.factory_policy(robot)
        return self.unit_policy(robot)

    def preplan_special_actions(self):
        planned = {}
        for robot in self.world.own.values():
            if robot.rtype == MINER and self.should_transform_miner(robot):
                planned[robot.uid] = ("TRANSFORM", robot.pos)
        return planned

    def commit(self, robot, action, next_pos):
        action, next_pos = self.normalize_action(robot, action)

        self.actions[robot.uid] = action

        if action in DIRS:
            self.reserved_next.add(next_pos)
        elif action.startswith("JUMP_"):
            self.reserved_next.add(next_pos)
        elif action in {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}:
            self.reserved_next.add(robot.pos)
            self.reserved_next.add(step_pos(robot.pos, "NORTH"))
        elif action == "TRANSFORM":
            # Unlike IDLE/TRANSFER/REMOVE, transform removes the miner before
            # movement. The cell is intentionally not reserved as occupied so
            # a carrier can step onto the newly created mine in the same turn.
            pass
        else:
            self.reserved_next.add(robot.pos)

    def normalize_action(self, robot, action):
        if action not in ACTION_SET:
            return "IDLE", robot.pos

        if action == "IDLE":
            return "IDLE", robot.pos

        # The engine pays upkeep before special actions and before movement
        # can matter. A robot that drops to 0 energy is forced idle, so every
        # non-idle proposal from policy code is shielded here.
        if not self.has_energy_after_upkeep(robot):
            return "IDLE", robot.pos

        if action in DIRS:
            if not action_ready(robot.move_cd):
                return "IDLE", robot.pos
            next_pos = step_pos(robot.pos, action)
            if not self.world.can_step(robot.pos, action, allow_unknown_target=True):
                return "IDLE", robot.pos
            if robot.rtype == FACTORY:
                if not self.factory_destination_safe(robot, next_pos):
                    return "IDLE", robot.pos
            elif not self.commit_destination_safe(robot, next_pos):
                return "IDLE", robot.pos
            return action, next_pos

        if action.startswith("JUMP_"):
            direction = action.split("_", 1)[1]
            emergency_jump = (
                robot.uid in self.emergency_jump_uids
                and EMERGENCY_JUMP_IGNORES_MOVE_CD
                and action_ready(robot.jump_cd)
            )
            if robot.rtype != FACTORY or direction not in DIRS or not (self.jump_ready(robot) or emergency_jump):
                return "IDLE", robot.pos
            landing = step_pos(robot.pos, direction, distance=2)
            if not self.world.in_bounds(landing):
                return "IDLE", robot.pos
            if not self.factory_destination_safe(robot, landing):
                return "IDLE", robot.pos
            return action, landing

        if action in {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}:
            if self.unit_build_legal(robot, action):
                return action, robot.pos
            return "IDLE", robot.pos

        if action.startswith("BUILD_"):
            if self.wall_build_legal(robot, action):
                return action, robot.pos
            return "IDLE", robot.pos

        if action.startswith("REMOVE_"):
            if self.wall_remove_legal(robot, action):
                return action, robot.pos
            return "IDLE", robot.pos

        if action.startswith("TRANSFER_"):
            if self.transfer_legal(robot, action):
                return action, robot.pos
            return "IDLE", robot.pos

        if action == "TRANSFORM":
            if self.transform_legal(robot):
                return action, robot.pos
            return "IDLE", robot.pos

        return "IDLE", robot.pos

    def unit_build_legal(self, robot, action):
        if robot.rtype != FACTORY or not action_ready(robot.build_cd):
            return False
        if robot.row - self.world.south < self.dynamic_danger_gap() + 2:
            return False
        unit_type = {
            "BUILD_SCOUT": SCOUT,
            "BUILD_WORKER": WORKER,
            "BUILD_MINER": MINER,
        }[action]
        cost = {
            SCOUT: int(cfg(self.world.config, "scoutCost", 50)),
            WORKER: int(cfg(self.world.config, "workerCost", 200)),
            MINER: int(cfg(self.world.config, "minerCost", 300)),
        }[unit_type]
        if not self.can_pay_after_upkeep(robot, cost):
            return False
        wall = self.world.current_wall(robot)
        spawn = step_pos(robot.pos, "NORTH")
        if wall is None or (wall & WALL_N) or not self.world.in_bounds(spawn):
            return False
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return False
        if not self.spawn_viable_for_unit(unit_type, spawn):
            return False
        return self.spawn_enemy_safe_for_type(unit_type, spawn, robot.owner)

    def wall_build_legal(self, robot, action):
        if robot.rtype != WORKER:
            return False
        direction = action.split("_", 1)[1]
        if direction not in DIRS:
            return False
        cost = int(cfg(self.world.config, "wallBuildCost", cfg(self.world.config, "wallRemoveCost", 100)))
        if not self.can_pay_after_upkeep(robot, cost):
            return False
        wall = self.world.current_wall(robot)
        if wall is None:
            return False
        target = step_pos(robot.pos, direction)
        return (
            not known_edge_blocked(self.world, robot.pos, direction)
            and self.world.in_bounds(target)
            and not is_fixed_wall(robot.pos, direction, self.world.width)
        )

    def wall_remove_legal(self, robot, action):
        if robot.rtype != WORKER:
            return False
        direction = action.split("_", 1)[1]
        if direction not in DIRS:
            return False
        cost = int(cfg(self.world.config, "wallRemoveCost", 100))
        if not self.can_pay_after_upkeep(robot, cost):
            return False
        wall = self.world.current_wall(robot)
        if wall is None:
            return False
        target = step_pos(robot.pos, direction)
        return (
            known_edge_blocked(self.world, robot.pos, direction)
            and self.world.in_bounds(target)
            and not is_fixed_wall(robot.pos, direction, self.world.width)
        )

    def transfer_legal(self, robot, action):
        if not self.has_energy_after_upkeep(robot):
            return False
        direction = action.split("_", 1)[1]
        if direction not in DIRS:
            return False
        if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
            return False
        return self.world.own_at(step_pos(robot.pos, direction)) is not None

    def transform_legal(self, robot):
        cost = int(cfg(self.world.config, "transformCost", 100))
        return (
            robot.rtype == MINER
            and self.can_pay_after_upkeep(robot, cost)
            and robot.pos in self.world.visible_nodes
            and robot.pos not in self.world.state["known_mines"]
        )

    def commit_destination_safe(self, robot, pos):
        forced_owner = self.forced_reserved_next.get(pos)
        if pos in self.reserved_next and forced_owner != robot.uid:
            return False
        occupant = self.world.own_at(pos)
        if occupant is not None and occupant.uid != robot.uid:
            if self.occupant_vacates_by_transform(occupant):
                self.forced_actions.setdefault(occupant.uid, ("TRANSFORM", occupant.pos))
            elif robot.rtype != FACTORY or occupant.rtype == FACTORY:
                return False
            elif (
                occupant.uid not in self.forced_actions
                and occupant.uid not in self.sacrifice_uids
            ):
                return False
        if crush_danger(robot, self.world.enemy_at(pos)):
            return False
        return True

    def evaluate_factory_destination(self, robot, pos):
        """Pure factory destination check.

        The returned decision describes the forced-vacate / sacrifice side
        effects that would be needed if this destination is selected. It does
        not mutate reservations, which keeps speculative candidate checks safe.
        """
        if pos in self.reserved_next:
            return SafetyDecision(False, {}, {}, set(), set(), "reserved")

        forced_actions = {}
        forced_reserved_next = {}
        reserved_next = set()
        sacrifice_uids = set()

        occupant = self.world.own_at(pos)
        if occupant is not None and occupant.uid != robot.uid:
            if occupant.rtype == FACTORY:
                return SafetyDecision(False, {}, {}, set(), set(), "factory_occupied")
            if self.occupant_vacates_by_transform(occupant):
                if occupant.uid not in self.forced_actions:
                    forced_actions[occupant.uid] = ("TRANSFORM", occupant.pos)
            elif (
                occupant.uid not in self.forced_actions
                and occupant.uid not in self.sacrifice_uids
            ):
                # Preserve useful units first. Sacrifice is reserved for true
                # factory tempo emergencies or depleted north-corridor blockers.
                vacate = self.find_vacate_action(occupant, blocked_positions={robot.pos, pos})
                if vacate is not None:
                    forced_actions[occupant.uid] = vacate
                    forced_reserved_next[vacate[1]] = occupant.uid
                    reserved_next.add(vacate[1])
                elif self.factory_sacrifice_allowed(robot, occupant, pos):
                    sacrifice_uids.add(occupant.uid)
                else:
                    return SafetyDecision(False, {}, {}, set(), set(), "blocked_by_friendly")
        if crush_danger(robot, self.world.enemy_at(pos)):
            return SafetyDecision(False, {}, {}, set(), set(), "enemy_crush")
        return SafetyDecision(True, forced_actions, forced_reserved_next, reserved_next, sacrifice_uids, "")

    def apply_factory_destination_decision(self, decision):
        self.forced_actions.update(decision.forced_actions)
        self.forced_reserved_next.update(decision.forced_reserved_next)
        self.reserved_next.update(decision.reserved_next)
        self.sacrifice_uids.update(decision.sacrifice_uids)

    def factory_destination_safe(self, robot, pos):
        """Selected factory destination check with side-effect application."""
        decision = self.evaluate_factory_destination(robot, pos)
        if not decision.ok:
            return False
        self.apply_factory_destination_decision(decision)
        return True

    def occupant_vacates_by_transform(self, occupant):
        # A miner on a profitable visible node can be treated like a planned
        # vacating blocker, because TRANSFORM destroys the robot before movement.
        # This is what allows same-turn harvest/factory follow-through.
        if occupant.rtype != MINER:
            return False
        forced = self.forced_actions.get(occupant.uid)
        if forced is not None:
            return forced[0] == "TRANSFORM"
        planned = self.planned_special_actions.get(occupant.uid)
        if planned is not None:
            return planned[0] == "TRANSFORM"
        if self.actions.get(occupant.uid) == "TRANSFORM":
            return True
        return False

    def factory_sacrifice_allowed(self, robot, occupant, pos):
        if robot.rtype != FACTORY or occupant.rtype == FACTORY or pos[1] < robot.row:
            return False
        gap = robot.row - self.world.south
        if gap <= FACTORY_SACRIFICE_GAP:
            return True
        if occupant.energy <= int(cfg(self.world.config, "energyPerTurn", 1)) and pos == step_pos(robot.pos, "NORTH"):
            return gap <= self.dynamic_danger_gap() + 2
        return False

    def find_vacate_action(self, robot, blocked_positions):
        if robot.energy <= int(cfg(self.world.config, "energyPerTurn", 1)):
            return None
        if not action_ready(robot.move_cd):
            return None
        reserved = set(self.reserved_next)
        reserved.update(blocked_positions)
        for direction, nxt in self.world.safe_moves(robot, reserved, prefer_north=True):
            if nxt not in blocked_positions:
                return direction, nxt
        return None

    def factory_policy(self, robot):
        wall = self.world.current_wall(robot)
        gap = robot.row - self.world.south
        danger = self.dynamic_danger_gap()
        emergency = gap <= danger

        route_quality = self.factory_route_quality(
            robot,
            emergency=emergency,
            apply=False,
        )
        floor = self.factory_floor_candidate(robot, wall, route_quality, emergency)
        if self.factory_floor_pressure(robot, route_quality):
            return floor

        complete = self.factory_complete_candidate(robot, wall, route_quality, emergency)
        if complete is not None:
            return complete
        return floor

    def factory_floor_pressure(self, robot, route_quality):
        return (
            route_quality.blocked
            or route_quality.margin_surplus <= FACTORY_FLOOR_PRESSURE_SURPLUS
            or self.front_worker_is_doing_wall_job(robot)
            or (
                route_quality.uses_jump
                and route_quality.margin_surplus <= FACTORY_FLOOR_PRESSURE_SURPLUS + 3
            )
        )

    def factory_floor_candidate(self, robot, wall, route_quality, emergency):
        emergency_jump = self.try_emergency_jump_north(robot)
        if emergency_jump is not None:
            return emergency_jump

        if emergency:
            jump = self.try_jump_over_north_wall(robot)
            if jump is not None:
                return jump

        if self.front_worker_is_doing_wall_job(robot):
            refuel = self.factory_refuel_worker(robot, route_quality, floor_mode=True)
            if refuel is not None:
                return refuel
            return "IDLE", robot.pos

        refuel = self.factory_refuel_worker(robot, route_quality, floor_mode=True)
        if refuel is not None:
            return refuel

        if route_quality.exists:
            if action_ready(robot.move_cd):
                return route_quality.action, route_quality.next_pos
            return "IDLE", robot.pos

        build = self.factory_build(robot, wall)
        if build is not None:
            return build

        if action_ready(robot.move_cd):
            action = self.factory_escape(robot, emergency=emergency)
            if action is not None:
                return action

        action = self.try_jump_north(robot)
        if action is not None:
            return action

        return "IDLE", robot.pos

    def factory_complete_candidate(self, robot, wall, route_quality, emergency):
        if route_quality.exists:
            if action_ready(robot.move_cd):
                return route_quality.action, route_quality.next_pos

            refuel = self.factory_refuel_worker(robot, route_quality, floor_mode=False)
            if refuel is not None:
                return refuel

            scout = self.factory_margin_scout_build(robot, wall, route_quality)
            if scout is not None:
                return scout

            if FACTORY_ROUTE_BLOCKS_BUILD:
                return "IDLE", robot.pos

        build = self.factory_build(robot, wall)
        if build is not None:
            return build

        if action_ready(robot.move_cd):
            action = self.factory_escape(robot, emergency=emergency)
            if action is not None:
                return action

        action = self.try_jump_north(robot)
        if action is not None:
            return action
        return None

    def dynamic_danger_gap(self):
        return dynamic_danger_gap_for_step(self.world.step, self.world.config)

    def try_emergency_jump_north(self, robot):
        gap = robot.row - self.world.south
        if gap > 2 or self.world.south <= 0:
            return None
        if not action_ready(robot.jump_cd):
            return None
        if REQUIRE_MOVE_READY_FOR_JUMP and not EMERGENCY_JUMP_IGNORES_MOVE_CD and not action_ready(robot.move_cd):
            return None
        landing = step_pos(robot.pos, "NORTH", distance=2)
        if self.jump_landing_safe(robot, landing):
            self.emergency_jump_uids.add(robot.uid)
            return "JUMP_NORTH", landing
        return None

    def try_jump_north(self, robot):
        if not self.jump_ready(robot):
            return None
        landing = step_pos(robot.pos, "NORTH", distance=2)
        if self.jump_landing_safe(robot, landing):
            return "JUMP_NORTH", landing
        return None

    def try_jump_over_north_wall(self, robot):
        wall = self.world.current_wall(robot)
        if wall is None or not (wall & WALL_N):
            return None
        if not self.jump_ready(robot):
            return None

        gap = robot.row - self.world.south
        danger = self.dynamic_danger_gap()
        pressured = gap <= danger + JUMP_ON_NORTH_WALL_GAP
        if not pressured:
            if self.has_adjacent_worker_wall_solution(robot):
                return None
            if self.world.step < JUMP_ON_NORTH_WALL_AFTER_STEP:
                return None

        landing = step_pos(robot.pos, "NORTH", distance=2)
        if self.jump_landing_safe(robot, landing):
            return "JUMP_NORTH", landing
        return None

    def jump_ready(self, robot):
        if not action_ready(robot.jump_cd):
            return False
        if REQUIRE_MOVE_READY_FOR_JUMP and not action_ready(robot.move_cd):
            return False
        return True

    def factory_refuel_worker(self, robot, route_quality=None, floor_mode=False):
        if not ENABLE_FACTORY_WORKER_REFUEL:
            return None
        if not self.has_energy_after_upkeep(robot):
            return None
        route_blocked = route_quality is None or route_quality.blocked
        critical_margin = (
            route_quality is not None
            and route_quality.margin_surplus <= FACTORY_FLOOR_PRESSURE_SURPLUS
        )
        pressure = floor_mode and (route_blocked or critical_margin)
        for direction in DIR_ORDER:
            other = self.world.own_at(step_pos(robot.pos, direction))
            if other is None or other.rtype != WORKER:
                continue
            if other.energy > FACTORY_WORKER_REFUEL_THRESHOLD:
                continue

            immediate_wall_job = worker_has_immediate_wall_job(self.world, other)
            route_critical = self.worker_is_route_critical(other)
            if not (immediate_wall_job or (pressure and route_critical)):
                continue

            if (
                not immediate_wall_job
                and robot.energy > FACTORY_WORKER_REFUEL_MAX_FACTORY_ENERGY
            ):
                continue

            room = MAX_ENERGY_BY_TYPE[WORKER] - other.energy
            if room <= 0:
                continue
            overflow = max(0, robot.energy - room)
            overflow_limit = (
                FACTORY_WORKER_REFUEL_PRESSURE_OVERFLOW
                if pressure or immediate_wall_job
                else FACTORY_WORKER_REFUEL_MAX_OVERFLOW
            )
            if overflow > overflow_limit:
                continue
            if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
                continue
            return "TRANSFER_" + direction, robot.pos
        return None

    def worker_is_route_critical(self, worker):
        factory = self.world.factory
        if factory is None or worker.rtype != WORKER:
            return False
        if manhattan(worker.pos, factory.pos) == 1 and worker.row >= factory.row:
            return True
        corridor = {
            (factory.col, factory.row + 1),
            (factory.col, factory.row + 2),
            (factory.col - 1, factory.row + 1),
            (factory.col + 1, factory.row + 1),
        }
        return worker.pos in corridor

    def can_spend_factory_tempo(self, robot, purpose, route_quality):
        if purpose == "scout_economy":
            return (
                ENABLE_MARGIN_SURPLUS_SCOUT
                and route_quality.exists
                and not route_quality.uses_jump
                and route_quality.north_gain > 0
                and route_quality.margin_surplus >= SCOUT_BUILD_MARGIN_SURPLUS
                and not action_ready(robot.move_cd)
            )
        return False

    def factory_margin_scout_build(self, robot, wall, route_quality):
        if not self.can_spend_factory_tempo(robot, "scout_economy", route_quality):
            return None
        if (
            wall is None
            or not action_ready(robot.build_cd)
            or self.counts.get(SCOUT, 0) >= self.scout_target_count(robot)
        ):
            return None
        spawn = step_pos(robot.pos, "NORTH")
        if wall & WALL_N or not self.world.in_bounds(spawn):
            return None
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return None
        scout_cost = int(cfg(self.world.config, "scoutCost", 50))
        if robot.energy < scout_cost + FACTORY_SCOUT_RESERVE:
            return None
        if not self.spawn_viable_for_unit(SCOUT, spawn):
            return None
        if not self.spawn_enemy_safe_for_type(SCOUT, spawn, robot.owner):
            return None
        self.counts[SCOUT] += 1
        return "BUILD_SCOUT", robot.pos

    def factory_step_candidate_pure(self, robot, direction, emergency):
        if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
            return None
        history = self.world.state.get("robot_history", {}).get(robot.uid, [])
        nxt = step_pos(robot.pos, direction)
        if direction != "NORTH" and not emergency and len(history) >= 2 and nxt in history[-6:]:
            return None
        if not self.factory_step_allowed(robot, direction, emergency):
            return None
        nxt = step_pos(robot.pos, direction)
        if self.factory_should_wait_for_occupant_job(nxt, emergency):
            return None
        if self.evaluate_factory_destination(robot, nxt).ok:
            return direction, nxt
        return None

    def factory_route_quality(self, robot, emergency=False, apply=False):
        candidate = self.factory_route_candidate(robot, emergency=emergency, apply=apply)
        if candidate is None:
            return RouteQuality(
                None,
                None,
                False,
                False,
                0,
                robot.row - self.world.south - self.dynamic_danger_gap(),
                True,
            )
        action, next_pos = candidate
        return RouteQuality(
            action,
            next_pos,
            True,
            action.startswith("JUMP_"),
            next_pos[1] - robot.row,
            robot.row - self.world.south - self.dynamic_danger_gap(),
            False,
        )

    def factory_route_candidate(self, robot, emergency=False, apply=False):
        north = step_pos(robot.pos, "NORTH")
        if self.world.own_at(north) is not None:
            direct = (
                self.factory_step_candidate(robot, "NORTH", emergency)
                if apply
                else self.factory_step_candidate_pure(robot, "NORTH", emergency)
            )
            if direct is not None:
                return direct

        target_row = min(self.world.north, robot.row + (8 if emergency else FACTORY_BFS_LOOKAHEAD))
        goals = [(col, target_row) for col in range(self.world.width)]
        step, _ = factory_optimistic_bfs_first_step(
            self.world,
            robot,
            robot.pos,
            goals,
            self.reserved_next,
            max_nodes=FACTORY_OPTIMISTIC_BFS_NODES,
            allow_friendly_future=False,
        )
        if step and step != "IDLE":
            candidate = (
                self.factory_step_candidate(robot, step, emergency)
                if apply
                else self.factory_step_candidate_pure(robot, step, emergency)
            )
            if candidate is not None:
                return candidate

        step, next_pos = factory_optimistic_jump_bfs_first_action(
            self.world,
            robot,
            robot.pos,
            goals,
            self.reserved_next,
            robot.jump_cd,
            max_nodes=FACTORY_JUMP_AWARE_BFS_NODES,
            allow_friendly_future=False,
        )
        if step and step != "IDLE":
            candidate = self.factory_route_action_candidate(
                robot, step, next_pos, emergency, apply
            )
            if candidate is not None:
                return candidate

        step, _ = factory_optimistic_bfs_first_step(
            self.world,
            robot,
            robot.pos,
            goals,
            self.reserved_next,
            max_nodes=FACTORY_OPTIMISTIC_BFS_NODES,
            allow_friendly_future=True,
        )
        if step and step != "IDLE":
            candidate = (
                self.factory_step_candidate(robot, step, emergency)
                if apply
                else self.factory_step_candidate_pure(robot, step, emergency)
            )
            if candidate is not None:
                return candidate

        step, next_pos = factory_optimistic_jump_bfs_first_action(
            self.world,
            robot,
            robot.pos,
            goals,
            self.reserved_next,
            robot.jump_cd,
            max_nodes=FACTORY_JUMP_AWARE_BFS_NODES,
            allow_friendly_future=True,
        )
        if step and step != "IDLE":
            candidate = self.factory_route_action_candidate(
                robot, step, next_pos, emergency, apply
            )
            if candidate is not None:
                return candidate

        return self.factory_jump_aware_route(robot, goals, apply=apply)

    def factory_route_action_candidate(self, robot, action, next_pos, emergency, apply):
        if action in DIRS:
            return (
                self.factory_step_candidate(robot, action, emergency)
                if apply
                else self.factory_step_candidate_pure(robot, action, emergency)
            )
        if not action.startswith("JUMP_"):
            return None
        if not next_pos or not self.world.in_bounds(next_pos):
            return None
        decision = self.evaluate_factory_destination(robot, next_pos)
        if not decision.ok:
            return None
        if apply:
            self.apply_factory_destination_decision(decision)
        return action, next_pos

    def factory_jump_aware_route(self, robot, goals, apply=False):
        if not self.jump_ready(robot):
            return None

        options = []
        for direction in ("NORTH", "EAST", "WEST"):
            landing = step_pos(robot.pos, direction, distance=2)
            if not self.world.in_bounds(landing):
                continue
            if crush_danger(robot, self.world.enemy_at(landing)):
                continue
            wall = self.world.wall_at(landing)
            if wall is not None and not any(
                self.world.can_step(landing, d, allow_unknown_target=True)
                for d in DIR_ORDER
            ):
                continue
            decision = self.evaluate_factory_destination(robot, landing)
            if not decision.ok:
                continue
            if landing in goals:
                reachable = True
            else:
                step, _ = factory_optimistic_bfs_first_step(
                    self.world,
                    robot,
                    landing,
                    goals,
                    self.reserved_next | {landing},
                    max_nodes=FACTORY_JUMP_AWARE_BFS_NODES,
                )
                reachable = step is not None
            if not reachable:
                continue
            score = (
                landing[1] - robot.row,
                direction == "NORTH",
                -abs(landing[0] - self.world.width // 2),
            )
            options.append((score, direction, landing, decision))

        if not options:
            return None
        options.sort(reverse=True)
        _, direction, landing, decision = options[0]
        if apply:
            self.apply_factory_destination_decision(decision)
        return "JUMP_" + direction, landing

    def factory_build(self, robot, wall):
        if wall is None or not action_ready(robot.build_cd):
            return None
        spawn = step_pos(robot.pos, "NORTH")
        if wall & WALL_N or not self.world.in_bounds(spawn):
            return None
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return None

        energy = robot.energy
        gap = robot.row - self.world.south
        scout_cost = int(cfg(self.world.config, "scoutCost", 50))
        miner_cost = int(cfg(self.world.config, "minerCost", 300))

        if self.opening_needs_worker(robot, wall):
            build = self.factory_build(robot, wall)
            if build is not None:
                return build

        scout = self.factory_opening_scout_build(robot, wall)
        if scout is not None:
            return scout

        build = self.factory_build(robot, wall)
        if build is not None:
            return build

        best_node = self.best_reachable_node_for_new_miner(spawn)
        if (
            best_node is not None
            and best_node[0] > MIN_MINER_BUILD_SCORE
            and self.counts.get(MINER, 0) < MINER_TARGET_COUNT
            and gap >= self.dynamic_danger_gap() + MINER_BUILD_MARGIN
            and energy >= miner_cost + FACTORY_MINER_RESERVE
            and self.spawn_viable_for_unit(MINER, spawn)
            and self.spawn_enemy_safe_for_type(MINER, spawn, robot.owner)
        ):
            self.counts[MINER] += 1
            return "BUILD_MINER", robot.pos

        scout_target = self.scout_target_count(robot)
        if (
            self.counts.get(SCOUT, 0) < scout_target
            and energy >= scout_cost + FACTORY_SCOUT_RESERVE
            and self.spawn_viable_for_unit(SCOUT, spawn)
            and self.spawn_enemy_safe_for_type(SCOUT, spawn, robot.owner)
        ):
            self.counts[SCOUT] += 1
            return "BUILD_SCOUT", robot.pos

        if (
            ALLOW_EXTRA_SCOUT
            and self.counts.get(SCOUT, 0) < scout_target + 1
            and energy >= scout_cost + FACTORY_GENERAL_RESERVE
            and 80 < self.world.step < 300
            and self.spawn_viable_for_unit(SCOUT, spawn)
            and self.spawn_enemy_safe_for_type(SCOUT, spawn, robot.owner)
        ):
            self.counts[SCOUT] += 1
            return "BUILD_SCOUT", robot.pos
        return None

    def scout_target_count(self, factory):
        return scout_target_count_for_world(self.world, factory)

    def factory_opening_scout_build(self, robot, wall):
        if (
            wall is None
            or not action_ready(robot.build_cd)
            or self.world.step >= OPENING_SCOUT_STEP_LIMIT
            or self.counts.get(SCOUT, 0) >= self.scout_target_count(robot)
        ):
            return None
        spawn = step_pos(robot.pos, "NORTH")
        if wall & WALL_N or not self.world.in_bounds(spawn):
            return None
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return None
        if self.opening_needs_worker(robot, wall):
            return None
        scout_cost = int(cfg(self.world.config, "scoutCost", 50))
        if robot.energy < scout_cost + FACTORY_SCOUT_RESERVE:
            return None
        if not self.spawn_viable_for_unit(SCOUT, spawn):
            return None
        if not self.spawn_enemy_safe_for_type(SCOUT, spawn, robot.owner):
            return None
        self.counts[SCOUT] += 1
        return "BUILD_SCOUT", robot.pos

    def opening_needs_worker(self, robot, wall):
        if self.counts.get(WORKER, 0) >= WORKER_TARGET_COUNT:
            return False
        if wall is not None and (wall & WALL_N):
            return True

        spawn = step_pos(robot.pos, "NORTH")
        if not self.world.in_bounds(spawn):
            return False
        spawn_wall = self.world.wall_at(spawn)
        if spawn_wall is None:
            return False

        exits = 0
        for direction in ("NORTH", "EAST", "WEST"):
            _, _, bit = DIRS[direction]
            nxt = step_pos(spawn, direction)
            if self.world.in_bounds(nxt) and not (spawn_wall & bit):
                exits += 1
        return exits <= OPENING_WORKER_BRANCH_EXIT_LIMIT

    def has_adjacent_worker_wall_solution(self, robot):
        for other in self.world.own.values():
            if other.rtype != WORKER:
                continue
            if manhattan(other.pos, robot.pos) > 2:
                continue
            if worker_has_immediate_wall_job(self.world, other):
                return True
        return False

    def front_worker_is_doing_wall_job(self, robot):
        front = step_pos(robot.pos, "NORTH")
        occupant = self.world.own_at(front)
        return (
            occupant is not None
            and occupant.rtype == WORKER
            and worker_has_immediate_wall_job(self.world, occupant)
        )

    def best_reachable_node_for_new_miner(self, spawn):
        if not self.world.in_bounds(spawn):
            return None
        probe = Robot(
            "probe-miner",
            MINER,
            spawn[0],
            spawn[1],
            int(cfg(self.world.config, "minerCost", 300)),
            self.world.player,
        )
        best = None
        transform_cost = int(cfg(self.world.config, "transformCost", 100))
        mine_rate = int(cfg(self.world.config, "mineRate", 50))
        for node in self.world.state["known_nodes"]:
            if node in self.world.state["known_mines"]:
                continue
            if not node_lifetime_viable(self.world, node):
                continue
            step, _, dist = bfs_first_step_and_distance(
                self.world,
                probe,
                [node],
                self.reserved_next,
                max_nodes=160,
                allow_occupied_goal=False,
            )
            if step is None:
                continue
            # A newly built miner cannot receive an action from this turn's
            # action dictionary, so add one turn before its first planned step.
            arrival_time = 1 + estimated_arrival_time(probe, dist, self.world.config)
            lifetime = expected_turns_until_row_scrolled(self.world, node[1])
            if lifetime < arrival_time + MIN_MINE_LIFETIME_TURNS:
                continue
            score = mine_rate * max(0, lifetime - arrival_time) - transform_cost - arrival_time
            if best is None or score > best[0]:
                best = (score, node, arrival_time)
        return best

    def factory_worker_build(self, robot, wall):
        if wall is None or not action_ready(robot.build_cd):
            return None
        spawn = step_pos(robot.pos, "NORTH")
        if wall & WALL_N or not self.world.in_bounds(spawn):
            return None
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return None
        worker_cost = int(cfg(self.world.config, "workerCost", 200))
        if (
            self.counts.get(WORKER, 0) < WORKER_TARGET_COUNT
            and robot.energy >= worker_cost + FACTORY_WORKER_RESERVE
            and self.spawn_viable_for_unit(WORKER, spawn)
            and self.spawn_enemy_safe_for_type(WORKER, spawn, robot.owner)
        ):
            self.counts[WORKER] += 1
            return "BUILD_WORKER", robot.pos
        return None

    def factory_corridor_worker_build(self, robot, wall):
        if (
            wall is None
            or not action_ready(robot.build_cd)
            or self.counts.get(WORKER, 0) >= WORKER_TARGET_COUNT
        ):
            return None
        spawn = step_pos(robot.pos, "NORTH")
        if wall & WALL_N or not self.world.in_bounds(spawn):
            return None
        if spawn in self.reserved_next or self.world.own_at(spawn) is not None:
            return None
        spawn_wall = self.world.wall_at(spawn)
        if spawn_wall is None or not (spawn_wall & WALL_N):
            return None
        if not self.spawn_enemy_safe_for_type(WORKER, spawn, robot.owner):
            return None
        worker_cost = int(cfg(self.world.config, "workerCost", 200))
        if robot.energy < worker_cost + FACTORY_WORKER_RESERVE:
            return None
        if not self.spawn_viable_for_unit(WORKER, spawn):
            return None
        self.counts[WORKER] += 1
        return "BUILD_WORKER", robot.pos

    def spawn_enemy_safe_for_type(self, unit_type, spawn, owner):
        enemy = self.world.enemy_at(spawn)
        if enemy is None:
            return True
        probe = Robot("spawn", unit_type, spawn[0], spawn[1], 1, owner)
        return not crush_danger(probe, enemy)

    def spawn_viable_for_unit(self, unit_type, spawn):
        spawn_wall = self.world.wall_at(spawn)
        if spawn_wall is None:
            return True
        if unit_type == WORKER:
            for direction in ("NORTH", "EAST", "WEST"):
                _, _, bit = DIRS[direction]
                nxt = step_pos(spawn, direction)
                if self.world.in_bounds(nxt) and (spawn_wall & bit):
                    return True
        return self.spawn_has_forward_exit(spawn)

    def spawn_has_forward_exit(self, spawn):
        spawn_wall = self.world.wall_at(spawn)
        if spawn_wall is None:
            return True
        for direction in ("NORTH", "EAST", "WEST"):
            _, _, bit = DIRS[direction]
            if spawn_wall & bit:
                continue
            nxt = step_pos(spawn, direction)
            if self.world.in_bounds(nxt) and self.world.own_at(nxt) is None:
                return True
        return False

    def factory_escape(self, robot, emergency):
        if not action_ready(robot.move_cd):
            return None

        direct = self.factory_step_candidate(robot, "NORTH", emergency)
        if direct is not None:
            return direct

        jump = self.try_jump_over_north_wall(robot)
        if jump is not None:
            return jump

        # Simple-ferry profile: use a long north horizon. The factory looks far
        # north and treats unknown future cells
        # optimistically, but the first committed step still passes the normal
        # safety shield.
        target_row = min(self.world.north, robot.row + (8 if emergency else FACTORY_BFS_LOOKAHEAD))
        goals = [(col, target_row) for col in range(self.world.width)]
        step, _ = factory_optimistic_bfs_first_step(
            self.world,
            robot,
            robot.pos,
            goals,
            self.reserved_next,
            max_nodes=FACTORY_OPTIMISTIC_BFS_NODES,
        )
        if step and step != "IDLE":
            nxt = step_pos(robot.pos, step)
            if (
                self.factory_step_allowed(robot, step, emergency)
                and not self.factory_should_wait_for_occupant_job(nxt, emergency)
                and self.factory_destination_safe(robot, nxt)
            ):
                return step, nxt

        jump_bfs = self.factory_jump_aware_escape(robot, goals)
        if jump_bfs is not None:
            return jump_bfs

        for direction, nxt in self.factory_move_options(robot, prefer_north=True):
            if (
                self.factory_step_allowed(robot, direction, emergency)
                and not self.factory_should_wait_for_occupant_job(nxt, emergency)
                and self.factory_destination_safe(robot, nxt)
            ):
                return direction, nxt

        if emergency and self.jump_ready(robot):
            landing = step_pos(robot.pos, "NORTH", distance=2)
            if self.jump_landing_safe(robot, landing):
                return "JUMP_NORTH", landing
        return None

    def factory_jump_aware_escape(self, robot, goals):
        if not self.jump_ready(robot):
            return None

        options = []
        for direction in ("NORTH", "EAST", "WEST"):
            landing = step_pos(robot.pos, direction, distance=2)
            if not self.world.in_bounds(landing):
                continue
            if crush_danger(robot, self.world.enemy_at(landing)):
                continue
            wall = self.world.wall_at(landing)
            if wall is not None and not any(
                self.world.can_step(landing, d, allow_unknown_target=True)
                for d in DIR_ORDER
            ):
                continue
            decision = self.evaluate_factory_destination(robot, landing)
            if not decision.ok:
                continue
            if landing in goals:
                reachable = True
            else:
                step, _ = factory_optimistic_bfs_first_step(
                    self.world,
                    robot,
                    landing,
                    goals,
                    self.reserved_next | {landing},
                    max_nodes=FACTORY_JUMP_AWARE_BFS_NODES,
                )
                reachable = step is not None
            if not reachable:
                continue
            score = (
                landing[1] - robot.row,
                direction == "NORTH",
                -abs(landing[0] - self.world.width // 2),
            )
            options.append((score, direction, landing, decision))

        if not options:
            return None
        options.sort(reverse=True)
        _, direction, landing, decision = options[0]
        self.apply_factory_destination_decision(decision)
        return "JUMP_" + direction, landing

    def factory_step_candidate(self, robot, direction, emergency):
        if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
            return None
        history = self.world.state.get("robot_history", {}).get(robot.uid, [])
        nxt = step_pos(robot.pos, direction)
        if direction != "NORTH" and not emergency and len(history) >= 2 and nxt in history[-6:]:
            return None
        if not self.factory_step_allowed(robot, direction, emergency):
            return None
        nxt = step_pos(robot.pos, direction)
        if self.factory_should_wait_for_occupant_job(nxt, emergency):
            return None
        if self.factory_destination_safe(robot, nxt):
            return direction, nxt
        return None

    def factory_should_wait_for_occupant_job(self, pos, emergency):
        if emergency:
            return False
        occupant = self.world.own_at(pos)
        if occupant is None or occupant.rtype == FACTORY:
            return False
        if (
            occupant.rtype == SCOUT
            and self.should_unload_scout(occupant)
            and self.transfer_to_factory_if_adjacent(occupant) is not None
        ):
            return True
        if occupant.rtype == WORKER and worker_has_immediate_wall_job(self.world, occupant):
            return True
        return False

    def factory_move_options(self, robot, prefer_north=True):
        options = []
        for direction, nxt in self.world.neighbors(robot.pos, allow_unknown_target=True):
            progress = nxt[1] - robot.row
            center_pull = -abs(nxt[0] - self.world.width // 2)
            if prefer_north:
                score = (progress, center_pull, direction == "NORTH")
            else:
                score = (center_pull, progress, direction == "NORTH")
            options.append((score, direction, nxt))
        options.sort(reverse=True)
        return [(direction, nxt) for _, direction, nxt in options]

    def factory_step_allowed(self, robot, direction, emergency):
        if direction != "SOUTH":
            return True
        if emergency:
            return False
        return robot.row - self.world.south > self.dynamic_danger_gap() + 5

    def jump_landing_safe(self, robot, landing):
        if not self.world.in_bounds(landing):
            return False
        if crush_danger(robot, self.world.enemy_at(landing)):
            return False
        wall = self.world.wall_at(landing)
        if wall is not None and not any(
            self.world.can_step(landing, direction, allow_unknown_target=True)
            for direction in DIR_ORDER
        ):
            return False
        return self.factory_destination_safe(robot, landing)

    def ordered_targets(self, robot, candidates):
        if not candidates:
            return candidates
        assigned = self.world.state["assigned_targets"].get(robot.uid)
        if assigned is None:
            return candidates
        try:
            old_kind, old_pos, old_step, _ = assigned
            old_pos = tuple(old_pos)
        except Exception:
            return candidates
        if self.world.step - int(old_step) > ASSIGNMENT_TTL:
            return candidates
        best_score = candidates[0].score
        for idx, target in enumerate(candidates):
            if target.kind == old_kind and target.pos == old_pos:
                # Mild hysteresis reduces oscillation. A unit should not abandon
                # a nearly-as-good target every time small score noise appears.
                if target.score >= best_score - ASSIGNMENT_SWITCH_MARGIN:
                    return [target] + candidates[:idx] + candidates[idx + 1 :]
                break
        return candidates

    def target_allowed_under_planned_ledger(self, robot, target):
        if target.kind != "unload_factory":
            return True
        if robot.rtype == SCOUT:
            return self.should_unload_scout(robot)
        if robot.rtype in (WORKER, MINER):
            return self.should_unload_non_scout(robot)
        return False

    def remember_target(self, robot, target):
        self.reserved_targets.add(target.pos)
        self.world.state["assigned_targets"][robot.uid] = (
            target.kind,
            target.pos,
            self.world.step,
            target.score,
        )

    def unit_policy(self, robot):
        if robot.energy <= int(cfg(self.world.config, "energyPerTurn", 1)):
            return "IDLE", robot.pos

        if robot.pos in self.reserved_next and robot.uid not in self.sacrifice_uids:
            # Another robot, usually the factory, is entering this cell. If this
            # unit can move away safely, preserving both units beats collision.
            vacate = self.find_vacate_action(robot, blocked_positions={robot.pos})
            if vacate is not None:
                return vacate
            return "IDLE", robot.pos

        if action_ready(robot.move_cd) and (
            robot.rtype != WORKER or not self.worker_has_wall_job(robot)
        ):
            # A neighboring miner may be committed to TRANSFORM before this
            # unit moves. Stepping onto that new mine can recover the deposit
            # immediately, so it is allowed to beat routine unload/transfer.
            transform_harvest = self.step_into_transforming_mine(robot)
            if transform_harvest is not None:
                return transform_harvest

        special = self.special_action(robot)
        if special is not None:
            return special

        if not action_ready(robot.move_cd):
            return "IDLE", robot.pos

        transform_harvest = self.step_into_transforming_mine(robot)
        if transform_harvest is not None:
            return transform_harvest

        candidates = select_candidate_targets(self.world, robot, self.reserved_targets)
        for target in self.ordered_targets(robot, candidates):
            if not self.target_allowed_under_planned_ledger(robot, target):
                continue
            if target.kind == "frontier" and target.pos == robot.pos:
                direction, nxt = self.world.frontier_exit(robot, self.reserved_next)
                if direction:
                    self.remember_target(robot, target)
                    return direction, nxt

            if target.kind in {"wall_open", "factory_corridor_open"} and robot.rtype == WORKER:
                remove = self.worker_remove_relevant_wall(
                    robot,
                    allow_default=(target.kind == "wall_open"),
                )
                if remove is not None:
                    self.remember_target(robot, target)
                    return remove

            if target.pos == robot.pos and target.kind in {"mine_harvest", "crystal", "unload_factory"}:
                self.remember_target(robot, target)
                return "IDLE", robot.pos

            if target.pos != robot.pos:
                step, _ = bfs_first_step(
                    self.world,
                    robot,
                    [target.pos],
                    self.reserved_next,
                    max_nodes=MAX_BFS_NODES,
                )
                if step and step != "IDLE":
                    nxt = step_pos(robot.pos, step)
                    if self.commit_destination_safe(robot, nxt):
                        self.remember_target(robot, target)
                        return step, nxt

        for direction, nxt in self.world.safe_moves(robot, self.reserved_next, prefer_north=True):
            return direction, nxt

        return "IDLE", robot.pos

    def special_action(self, robot):
        # Ordering is deliberate. Transform and useful wall edits spend the
        # unit's energy on its job. All-energy transfer comes later because it
        # can turn a worker/miner into an immobile blocker.
        if robot.rtype == MINER and self.should_transform_miner(robot):
            return "TRANSFORM", robot.pos

        if robot.rtype == WORKER:
            remove = self.worker_remove_relevant_wall(robot, allow_default=False)
            if remove is not None:
                return remove

        if robot.rtype == SCOUT and self.should_unload_scout(robot):
            unload = self.transfer_to_factory_if_adjacent(robot)
            if unload is not None:
                return unload

        if robot.rtype in (WORKER, MINER) and self.should_unload_non_scout(robot):
            unload = self.transfer_to_factory_if_adjacent(robot)
            if unload is not None:
                return unload

        transfer = self.transfer_if_helpful(robot, allow_non_scout=self.should_unload_non_scout(robot))
        if transfer is not None:
            return transfer

        return None

    def should_unload_scout(self, robot):
        if robot.rtype != SCOUT:
            return False
        if robot.row - self.world.south <= self.dynamic_danger_gap() + 1:
            return True
        if self.is_factory_north_cell(robot) and self.world.factory is not None:
            factory_gap = self.world.factory.row - self.world.south
            if factory_gap <= self.dynamic_danger_gap() + 3:
                return False
        planned_build = self.factory_planned_build_action()
        if planned_build is not None:
            # Transfer is processed after factory build. Once a build has
            # already been selected, same-turn scout liquidation cannot fund
            # it, and the next build is delayed by cooldown. Preserve scout
            # mobility unless the scroll-danger branch above already fired.
            return False
        if robot.energy >= SCOUT_FORCE_RETURN_ENERGY:
            return True
        if robot.energy >= SCOUT_RETURN_ENERGY and not scout_has_good_forward_job(self.world, robot):
            return True
        if not wants_to_unload(robot):
            return False
        if self.transfer_crosses_build_threshold(robot):
            return True
        cap = MAX_ENERGY_BY_TYPE.get(SCOUT, 100)
        if robot.energy >= cap - 2 and not scout_has_good_forward_job(self.world, robot):
            return True
        return False

    def step_into_transforming_mine(self, robot):
        if robot.rtype == MINER:
            return None
        for direction, nxt in self.world.neighbors(robot.pos, allow_unknown_target=True):
            occupant = self.world.own_at(nxt)
            if occupant is None or occupant.uid == robot.uid:
                continue
            if occupant.rtype != MINER:
                continue
            if not (
                self.actions.get(occupant.uid) == "TRANSFORM"
                or self.forced_actions.get(occupant.uid, (None, None))[0] == "TRANSFORM"
                or self.planned_special_actions.get(occupant.uid, (None, None))[0] == "TRANSFORM"
            ):
                continue
            if self.commit_destination_safe(robot, nxt):
                return direction, nxt
        return None

    def should_transform_miner(self, robot):
        if robot.pos not in self.world.visible_nodes:
            return False
        if robot.pos in self.world.state["known_mines"]:
            return False
        cost = int(cfg(self.world.config, "transformCost", 100))
        if robot.energy < cost + MINE_TRANSFORM_BUFFER:
            return False
        mine_rate = int(cfg(self.world.config, "mineRate", 50))
        lifetime = expected_turns_until_row_scrolled(self.world, robot.row)
        if lifetime < MIN_MINE_LIFETIME_TURNS:
            return False
        # A mine that cannot be harvested is mostly locked energy. Carrier time
        # is estimated with BFS reachability and unit move period, not just
        # Manhattan distance.
        carrier_time = self.nearest_harvest_carrier_time(robot)
        if carrier_time is None:
            if lifetime < UNATTENDED_MINE_MIN_LIFETIME_TURNS or self.world.step >= 300:
                return False
            recover_probability = 0.35
            effective_carrier_time = lifetime * 0.5
        else:
            recover_probability = 1.0 if carrier_time <= 3 else 0.7
            effective_carrier_time = carrier_time

        # Transform moves the miner's remaining energy into the mine. That
        # deposit is not robot-score energy again until a friendly carrier
        # harvests it, so low-recoverability mines are penalized even when their
        # future generation looks positive.
        deposit = max(0, robot.energy - self.energy_per_turn() - cost)
        expected_harvest_turns = max(0, lifetime - effective_carrier_time)
        expected_harvested = mine_rate * expected_harvest_turns
        score = (
            recover_probability * (deposit + expected_harvested)
            - deposit
            - cost
            - effective_carrier_time
            - MINE_TRANSFORM_LOST_UNIT_PENALTY
        )
        return score > 0

    def nearest_harvest_carrier_distance(self, robot):
        best = None
        for other in self.world.own.values():
            if other.uid == robot.uid:
                continue
            if other.energy <= int(cfg(self.world.config, "energyPerTurn", 1)):
                continue
            if other.rtype != FACTORY:
                cap = MAX_ENERGY_BY_TYPE.get(other.rtype, 0)
                if cap - other.energy < 50:
                    continue
            step, _, dist = bfs_first_step_and_distance(
                self.world,
                other,
                [robot.pos],
                self.reserved_next,
                max_nodes=120,
                allow_occupied_goal=True,
            )
            if step is None:
                continue
            if best is None or dist < best:
                best = dist
        return best

    def nearest_harvest_carrier_time(self, robot):
        best = None
        for other in self.world.own.values():
            if other.uid == robot.uid:
                continue
            if other.energy <= int(cfg(self.world.config, "energyPerTurn", 1)):
                continue
            if other.rtype != FACTORY:
                cap = MAX_ENERGY_BY_TYPE.get(other.rtype, 0)
                if cap - other.energy < 50:
                    continue
            step, _, dist = bfs_first_step_and_distance(
                self.world,
                other,
                [robot.pos],
                self.reserved_next,
                max_nodes=120,
                allow_occupied_goal=True,
            )
            if step is None:
                continue
            arrival_time = estimated_arrival_time(other, dist, self.world.config)
            if best is None or arrival_time < best:
                best = arrival_time
        return best

    def worker_remove_relevant_wall(self, robot, allow_default=False):
        wall = self.world.current_wall(robot)
        if wall is None:
            return None
        cost = int(cfg(self.world.config, "wallRemoveCost", 100))
        if robot.energy < cost + WALL_REMOVE_BUFFER:
            return None

        def can_remove(direction):
            nxt = step_pos(robot.pos, direction)
            return (
                known_edge_blocked(self.world, robot.pos, direction)
                and self.world.in_bounds(nxt)
                and not is_fixed_wall(robot.pos, direction, self.world.width)
            )

        factory = self.world.factory
        if factory is not None and manhattan(robot.pos, factory.pos) == 1:
            for direction in DIR_ORDER:
                if step_pos(robot.pos, direction) == factory.pos and can_remove(direction):
                    return "REMOVE_" + direction, robot.pos

        if factory is not None:
            corridor = {
                factory.pos,
                (factory.col, factory.row + 1),
                (factory.col - 1, factory.row + 1),
                (factory.col + 1, factory.row + 1),
            }
            if robot.pos in corridor and can_remove("NORTH"):
                return "REMOVE_NORTH", robot.pos
            for direction in DIR_ORDER:
                if step_pos(robot.pos, direction) in corridor and can_remove(direction):
                    return "REMOVE_" + direction, robot.pos

        if allow_default and can_remove("NORTH"):
            return "REMOVE_NORTH", robot.pos
        return None

    def worker_has_wall_job(self, robot):
        return robot.rtype == WORKER and self.worker_remove_relevant_wall(robot, allow_default=False) is not None

    def miner_has_node_or_mine_job(self, robot):
        if robot.rtype != MINER:
            return False
        if (
            robot.pos in self.world.visible_nodes
            and robot.pos not in self.world.state["known_mines"]
            and node_lifetime_viable(self.world, robot.pos)
        ):
            return True
        for node in self.world.state["known_nodes"]:
            if node in self.world.state["known_mines"]:
                continue
            if node_lifetime_viable(self.world, node):
                return True
        for mine_pos, mine_data in self.world.state["known_mines"].items():
            if not self.world.in_bounds(mine_pos):
                continue
            energy = estimated_mine_energy(self.world, mine_pos, mine_data)
            _, _, owner, _ = parse_mine_record(mine_data)
            try:
                owner = int(owner)
            except Exception:
                continue
            if (
                owner == self.world.player
                and energy >= MINE_HARVEST_MIN_ENERGY
                and node_lifetime_viable(self.world, mine_pos, min_turns=max(MINE_HARVEST_SCROLL_GAP, 4))
            ):
                return True
        return False

    def should_unload_non_scout(self, robot):
        if robot.rtype not in (WORKER, MINER) or not wants_to_unload(robot):
            return False
        danger = self.dynamic_danger_gap()
        if robot.row - self.world.south <= danger:
            return True
        # The north cell is the factory's main escape lane. Turning a worker or
        # miner there into a 0-energy body is usually worse than keeping its job
        # energy, unless scroll liquidation already applies above.
        if self.is_factory_north_cell(robot):
            return False
        if robot.rtype == WORKER and self.worker_has_wall_job(robot):
            return False
        if robot.rtype == MINER and self.miner_has_node_or_mine_job(robot):
            return False
        return self.transfer_crosses_build_threshold(robot)

    def is_factory_north_cell(self, robot):
        return self.world.factory is not None and robot.pos == step_pos(self.world.factory.pos, "NORTH")

    def pending_build_needs(self):
        return self.pending_build_needs_after_planned_actions()

    def transfer_crosses_build_threshold(self, robot):
        factory = self.world.factory
        if factory is None:
            return False
        # Factory plans first. If it already committed a build, a later transfer
        # cannot fund that build and the next build is delayed by cooldown, so
        # treating the static world state as "still needs a worker/scout/miner"
        # would over-liquidate carrier units.
        if self.factory_planned_build_action() is not None:
            return False
        needs = self.pending_build_needs_after_planned_actions()
        if not needs:
            return False
        ept = self.energy_per_turn()
        energy_for_next_build = factory.energy - ept + max(0, robot.energy - ept) - ept
        return any(factory.energy < need <= energy_for_next_build for need in needs)

    def factory_planned_build_action(self):
        factory = self.world.factory
        if factory is None:
            return None
        action = self.actions.get(factory.uid)
        if action in {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}:
            return action
        return None

    def pending_build_needs_after_planned_actions(self):
        factory = self.world.factory
        if factory is None or not action_ready(factory.build_cd):
            return []
        if self.factory_planned_build_action() is not None:
            return []

        # `self.counts` is updated when the factory policy chooses a build, so
        # this ledger sees planned production that is not yet present in
        # `world.own`. Transfer policy should reason over this planned state.
        needs = []
        if self.counts.get(WORKER, 0) < WORKER_TARGET_COUNT:
            needs.append(int(cfg(self.world.config, "workerCost", 200)) + FACTORY_WORKER_RESERVE)

        wall = self.world.current_wall(factory)
        spawn = step_pos(factory.pos, "NORTH")
        best_node = None
        if (
            wall is not None
            and not (wall & WALL_N)
            and self.world.in_bounds(spawn)
            and self.spawn_viable_for_unit(MINER, spawn)
        ):
            best_node = self.best_reachable_node_for_new_miner(spawn)
        if (
            best_node is not None
            and best_node[0] > MIN_MINER_BUILD_SCORE
            and self.counts.get(MINER, 0) < MINER_TARGET_COUNT
            and factory.row - self.world.south >= self.dynamic_danger_gap() + MINER_BUILD_MARGIN
        ):
            needs.append(int(cfg(self.world.config, "minerCost", 300)) + FACTORY_MINER_RESERVE)

        if self.counts.get(SCOUT, 0) < self.scout_target_count(factory):
            needs.append(int(cfg(self.world.config, "scoutCost", 50)) + FACTORY_SCOUT_RESERVE)

        return [need for need in needs if factory.energy < need]

    def transfer_to_factory_if_adjacent(self, robot):
        if robot.rtype == FACTORY or not wants_to_unload(robot):
            return None
        for direction in DIR_ORDER:
            if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
                continue
            other = self.world.own_at(step_pos(robot.pos, direction))
            if other is not None and other.rtype == FACTORY:
                return "TRANSFER_" + direction, robot.pos
        return None

    def transfer_if_helpful(self, robot, allow_non_scout=False):
        if robot.energy <= LOW_ENERGY_TRANSFER_THRESHOLD:
            return None
        if robot.rtype in (WORKER, MINER):
            if not allow_non_scout:
                return None
            for direction in DIR_ORDER:
                if not self.world.can_step(robot.pos, direction, allow_unknown_target=True):
                    continue
                other = self.world.own_at(step_pos(robot.pos, direction))
                if other is not None and other.rtype == FACTORY:
                    return "TRANSFER_" + direction, robot.pos
            return None

        return None
# ============================================================
# Agent Entry Point
# ============================================================


def compute_actions(obs, config, started, state_store=None, strategy_cls=None):
    """Build a world model, inject a strategy class, and return actions.

    `state_store` and `strategy_cls` are the experiment boundary. Local
    benchmarks, replay conversion, and future strategy variants can reuse the
    same world/physics/safety code without changing the Kaggle-facing `agent()`
    wrapper.
    """

    world = WorldModel(obs, config, state_store=state_store)
    act_timeout = cfg(world.config, "actTimeout", 1.0)
    soft_budget = max(0.05, min(SOFT_ACT_DEADLINE, float(act_timeout) * 0.75))
    deadline = started + soft_budget
    strategy_type = strategy_cls or Strategy
    strategy = strategy_type(world, started, deadline)
    return strategy.plan()


def agent(obs, config):
    started = time.perf_counter()
    try:
        return compute_actions(obs, config, started)
    except Exception:
        try:
            obs_plain = to_plain_dict(obs)
            player = int(obs_plain.get("player", 0))
            actions = {}
            for uid, data in (obs_plain.get("robots") or {}).items():
                try:
                    if int(data[4]) == player:
                        actions[str(uid)] = "IDLE"
                except Exception:
                    continue
            return actions
        except Exception:
            return {}


def act(obs, config):
    return agent(obs, config)


__all__ = ["agent", "act"]
