from kaggle_environments import make

def get_outcome(final):
    # Determine the status
    s0, s1 = final[0].status, final[1].status
    r0, r1 = final[0].reward, final[1].reward
    
    # Try to extract factory position from observations
    robots = final[0].observation.get("robots", {})
    p0_has_factory = any(d[0] == 0 and d[4] == 0 for d in robots.values())
    p1_has_factory = any(d[0] == 0 and d[4] == 1 for d in robots.values())
    
    if not p0_has_factory and not p1_has_factory:
        return f"Mutual factory destruction (collision or scroll). P0 reward={r0}, P1 reward={r1}"
    elif not p0_has_factory:
        return f"P0 eliminated (factory destroyed). P1 wins. P0 reward={r0}, P1 reward={r1}"
    elif not p1_has_factory:
        return f"P1 eliminated (factory destroyed). P0 wins. P0 reward={r0}, P1 reward={r1}"
    else:
        return f"Time limit reached. P0 reward={r0}, P1 reward={r1}"

# Let's run the same 10 rounds (20 matches)
for round_idx in range(10):
    seed = 42 + round_idx
    
    # Match 1: A=agent_lean.py (0), B=starter_bfs.py (1)
    env = make("crawl", configuration={"randomSeed": seed}, debug=False)
    env.run(["agent_lean.py", "starter_bfs.py"])
    final = env.steps[-1]
    outcome1 = get_outcome(final)
    print(f"Round {round_idx} (Seed {seed}) - Match 1: A (Player 0) vs B (Player 1) - Steps: {len(env.steps)} | {outcome1}")

    # Match 2: B=starter_bfs.py (0), A=agent_lean.py (1)
    env = make("crawl", configuration={"randomSeed": seed}, debug=False)
    env.run(["starter_bfs.py", "agent_lean.py"])
    final = env.steps[-1]
    outcome2 = get_outcome(final)
    print(f"Round {round_idx} (Seed {seed}) - Match 2: B (Player 0) vs A (Player 1) - Steps: {len(env.steps)} | {outcome2}")
