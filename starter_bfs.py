

# ========================================%%writefile submission.py
"""Crawl agent v2: scout-first planning, dead-end JUMP, anti-oscillation."""

from collections import deque

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
DIRS = ["NORTH", "EAST", "WEST", "SOUTH"]
OFFSETS = {
    "NORTH": (0, 1),
    "SOUTH": (0, -1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}
WALL_BITS = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}
MAX_ENERGY = {SCOUT: 100, WORKER: 300, MINER: 500}
CRUSH_RANK = {FACTORY: 4, MINER: 3, WORKER: 2, SCOUT: 1}

_STATE = {
    "known_mining_nodes": set(),
    "explored": set(),
    "last_factory_pos": None,
    "factory_stuck_count": 0,
}


def agent(obs, config):
    state = _STATE
    actions = {}
    width = config.width
    south = obs.southBound
    north = obs.northBound

    def in_bounds(col, row):
        return 0 <= col < width and south <= row <= north

    def wall_at(col, row):
        if not in_bounds(col, row):
            return 0
        idx = (row - south) * width + col
        if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
            return obs.walls[idx]
        return 0

    def is_known(col, row):
        if not in_bounds(col, row):
            return False
        idx = (row - south) * width + col
        return 0 <= idx < len(obs.walls) and obs.walls[idx] != -1

    def can_move(col, row, direction):
        dc, dr = OFFSETS[direction]
        ncol, nrow = col + dc, row + dr
        if not in_bounds(ncol, nrow):
            return False
        return not (wall_at(col, row) & WALL_BITS[direction])

    def can_move_known(col, row, direction):
        """Stricter: refuse to traverse into unknown cells (no magical fog paths)."""
        if not is_known(col, row):
            return False
        dc, dr = OFFSETS[direction]
        ncol, nrow = col + dc, row + dr
        if not in_bounds(ncol, nrow):
            return False
        if not is_known(ncol, nrow):
            return False
        return not (wall_at(col, row) & WALL_BITS[direction])

    my_robots = {uid: data for uid, data in obs.robots.items() if data[4] == obs.player}
    occupied = {(d[1], d[2]): (uid, d[0], d[4]) for uid, d in obs.robots.items()}

    for r in range(south, north + 1):
        for c in range(width):
            if is_known(c, r):
                state["explored"].add((c, r))
    for key in obs.miningNodes:
        cc, rr = map(int, key.split(","))
        state["known_mining_nodes"].add((cc, rr))
    for key in obs.mines:
        cc, rr = map(int, key.split(","))
        state["known_mining_nodes"].discard((cc, rr))
    state["known_mining_nodes"] = {(c, r) for (c, r) in state["known_mining_nodes"] if r >= south}

    crystals = {}
    for key, value in obs.crystals.items():
        cc, rr = map(int, key.split(","))
        crystals[(cc, rr)] = value

    mines = {}
    for key, val in obs.mines.items():
        cc, rr = map(int, key.split(","))
        mines[(cc, rr)] = val

    reserved = set()
    decided = {}

    def bfs_path(start, goals_set, max_depth=40, blocked_cells=None, known_only=False):
        if not goals_set or start in goals_set:
            return None
        blocked_cells = blocked_cells or set()
        cm = can_move_known if known_only else can_move
        q = deque([(start, None, 0)])
        seen = {start}
        while q:
            (col, row), first, dist = q.popleft()
            if dist > 0 and (col, row) in goals_set:
                return first
            if dist >= max_depth:
                continue
            for d in DIRS:
                if not cm(col, row, d):
                    continue
                dc, dr = OFFSETS[d]
                nxt = (col + dc, row + dr)
                if nxt in seen:
                    continue
                if nxt in blocked_cells and nxt not in goals_set:
                    continue
                seen.add(nxt)
                q.append((nxt, first if first else d, dist + 1))
        return None

    def is_safe_move(uid, my_type, target):
        if target in reserved:
            return False
        if target in occupied:
            o_uid, o_type, o_owner = occupied[target]
            if o_uid == uid:
                return True
            if o_owner == obs.player:
                their_action = decided.get(o_uid)
                if their_action and their_action in OFFSETS:
                    return True  # friendly is leaving
                if their_action is None:
                    # OPTIMISTIC: friendly hasn't planned yet, assume scouts move
                    if o_type == SCOUT:
                        return True
                    return False
                return False
            # enemy: only safe if we strictly outrank
            return CRUSH_RANK[my_type] > CRUSH_RANK[o_type]
        return True

    def commit(uid, action, blocked_cell=None):
        decided[uid] = action
        actions[uid] = action
        col, row = my_robots[uid][1], my_robots[uid][2]
        if action in OFFSETS:
            dc, dr = OFFSETS[action]
            reserved.add((col + dc, row + dr))
        elif action.startswith("JUMP_"):
            d = action.split("_")[1]
            dc, dr = OFFSETS[d]
            reserved.add((col + 2 * dc, row + 2 * dr))
        else:
            reserved.add((col, row))
        if blocked_cell:
            reserved.add(blocked_cell)

    counts = {t: 0 for t in (FACTORY, SCOUT, WORKER, MINER)}
    for d in my_robots.values():
        counts[d[0]] += 1

    # PLAN ORDER: scouts FIRST (they get out of the way), factory LAST
    type_priority = {SCOUT: 0, WORKER: 1, MINER: 2, FACTORY: 3}
    ordered = sorted(my_robots.items(), key=lambda kv: (type_priority[kv[1][0]], kv[0]))

    for uid, data in ordered:
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        move_cd = data[5] if len(data) > 5 else 0
        jump_cd = data[6] if len(data) > 6 else 0
        build_cd = data[7] if len(data) > 7 else 0
        walls = wall_at(col, row)
        gap_from_south = row - south

        # ============ FACTORY ============
        if rtype == FACTORY:
            critical = gap_from_south <= 2 and south > 0
            blocked_north = bool(walls & WALL_BITS["NORTH"])
            spawn_cell = (col, row + 1)
            spawn_clear = (
                build_cd <= 1
                and row + 1 <= north
                and not blocked_north
                and spawn_cell not in occupied
                and spawn_cell not in reserved
            )

            need_build = None
            if spawn_clear and not critical and state["factory_stuck_count"] < 2:
                if counts[SCOUT] < 1 and energy >= config.scoutCost + 600 and gap_from_south >= 2:
                    need_build = "BUILD_SCOUT"
                elif (counts[MINER] < 1
                      and any(n[1] >= south for n in state["known_mining_nodes"])
                      and energy >= config.minerCost + 500):
                    need_build = "BUILD_MINER"
                elif counts[WORKER] < 1 and energy >= config.workerCost + 500:
                    has_blocking_walls = False
                    for r2 in range(row, min(row + 5, north + 1)):
                        for c2 in range(max(0, col - 2), min(width, col + 3)):
                            if is_known(c2, r2) and wall_at(c2, r2) & WALL_BITS["NORTH"]:
                                has_blocking_walls = True
                                break
                        if has_blocking_walls:
                            break
                    if has_blocking_walls:
                        need_build = "BUILD_WORKER"
                elif counts[SCOUT] < 2 and energy >= config.scoutCost + 800:
                    need_build = "BUILD_SCOUT"

            if need_build:
                commit(uid, need_build, spawn_cell)
                if "SCOUT" in need_build:  counts[SCOUT]  += 1
                if "WORKER" in need_build: counts[WORKER] += 1
                if "MINER" in need_build:  counts[MINER]  += 1
                continue

            if move_cd <= 1:
                target_row = min(north, row + 8)
                # Goals: KNOWN cells at row+2 or higher (no fog goals)
                goals = set()
                for r in range(row + 2, target_row + 1):
                    for c in range(width):
                        if is_known(c, r):
                            goals.add((c, r))
                # If nothing known forward, fall back to any cell with row > current
                if not goals:
                    for r in range(row + 1, target_row + 1):
                        for c in range(width):
                            if is_known(c, r):
                                goals.add((c, r))

                # BFS-block: friendly cells, BUT only if they're not committed to leaving
                blocked = set()
                for p, (o_uid, o_type, o_owner) in occupied.items():
                    if p == (col, row):
                        continue
                    if o_owner == obs.player and decided.get(o_uid) in OFFSETS:
                        continue
                    blocked.add(p)

                last_pos = state.get("last_factory_pos")
                # Use known-only BFS — no magical fog paths
                step = bfs_path((col, row), goals, max_depth=40, blocked_cells=blocked, known_only=True)

                def safe_neighbors(avoid_last_pos=True, allow_south=False):
                    out = []
                    pref = ["NORTH", "EAST", "WEST"]
                    if allow_south or critical:
                        pref.append("SOUTH")
                    for d in pref:
                        if not can_move(col, row, d):
                            continue
                        tgt = (col + OFFSETS[d][0], row + OFFSETS[d][1])
                        if avoid_last_pos and tgt == last_pos:
                            continue
                        if is_safe_move(uid, FACTORY, tgt):
                            out.append((d, tgt))
                    return out

                # Try BFS step ONLY if it's a forward direction (N/E/W) and not backward
                if step and step != "SOUTH":
                    tgt = (col + OFFSETS[step][0], row + OFFSETS[step][1])
                    going_backward = (tgt == last_pos and not critical)
                    if not going_backward and is_safe_move(uid, FACTORY, tgt):
                        state["last_factory_pos"] = (col, row)
                        state["factory_stuck_count"] = 0
                        commit(uid, step)
                        continue

                # Direct safe neighbors (prefer N)
                neighbors = safe_neighbors(avoid_last_pos=True)
                if neighbors:
                    d, _ = neighbors[0]
                    state["last_factory_pos"] = (col, row)
                    state["factory_stuck_count"] = 0
                    commit(uid, d)
                    continue

                # No safe forward move — try JUMP_NORTH (this is the dead-end escape)
                if (jump_cd <= 1 and row + 2 <= north
                    and (col, row + 2) not in occupied
                    and (col, row + 2) not in reserved):
                    state["last_factory_pos"] = (col, row)
                    state["factory_stuck_count"] = 0
                    commit(uid, "JUMP_NORTH")
                    continue

                # Allow stepping back to last_pos if otherwise stuck
                neighbors_back = safe_neighbors(avoid_last_pos=False)
                if neighbors_back:
                    d, _ = neighbors_back[0]
                    state["last_factory_pos"] = (col, row)
                    state["factory_stuck_count"] = 0
                    commit(uid, d)
                    continue

                # Truly stuck — IDLE
                state["factory_stuck_count"] = state.get("factory_stuck_count", 0) + 1
                commit(uid, "IDLE")
                continue

            commit(uid, "IDLE")
            continue

        # ============ non-factory ============
        if move_cd > 1:
            commit(uid, "IDLE")
            continue

        on_node = (col, row) in state["known_mining_nodes"]

        if rtype == MINER and on_node and energy >= config.transformCost + 50:
            commit(uid, "TRANSFORM")
            continue

        if rtype == WORKER and (walls & WALL_BITS["NORTH"]) and energy >= config.wallRemoveCost + 50:
            if row + 1 <= north:
                commit(uid, "REMOVE_NORTH")
                continue

        max_e = MAX_ENERGY.get(rtype, 10**9)
        want_energy = (max_e - energy) > 5

        if rtype == MINER:
            goals = {n for n in state["known_mining_nodes"] if n[1] >= south}
        elif rtype == SCOUT:
            if want_energy and crystals:
                goals = set(crystals.keys())
            else:
                tr = min(north, row + 6)
                goals = {(c, tr) for c in range(width)}
                for r2 in range(south, north + 1):
                    for c2 in range(width):
                        if (c2, r2) not in state["explored"] and abs(c2 - col) + abs(r2 - row) <= 12:
                            goals.add((c2, r2))
        else:  # WORKER
            if want_energy and crystals:
                goals = set(crystals.keys())
            else:
                tr = min(north, row + 4)
                goals = {(c, tr) for c in range(width)}

        blocked = {p for p in occupied if p != (col, row)}
        step = bfs_path((col, row), goals, max_depth=20, blocked_cells=blocked)

        if step:
            tgt = (col + OFFSETS[step][0], row + OFFSETS[step][1])
            if is_safe_move(uid, rtype, tgt):
                commit(uid, step)
                continue

        committed = False
        for d in ["NORTH", "EAST", "WEST", "SOUTH"]:
            if can_move(col, row, d):
                tgt = (col + OFFSETS[d][0], row + OFFSETS[d][1])
                if is_safe_move(uid, rtype, tgt):
                    commit(uid, d)
                    committed = True
                    break
        if not committed:
            commit(uid, "IDLE")

    return actions
