import sys
import importlib
from kaggle_environments import make

def run_benchmark(agent_a_path, agent_b_path, num_rounds=10):
    env = make("crawl", debug=False)
    
    results = {
        "A_wins": 0,
        "B_wins": 0,
        "draws": 0,
        "A_errors": 0,
        "B_errors": 0,
        "A_avg_reward": 0.0,
        "B_avg_reward": 0.0
    }
    
    # We will play num_rounds twice (swapping sides)
    for round_idx in range(num_rounds):
        # Seed
        seed = 42 + round_idx
        
        # Match 1: A is player 0, B is player 1
        env = make("crawl", configuration={"randomSeed": seed}, debug=False)
        env.run([agent_a_path, agent_b_path])
        final = env.steps[-1]
        
        r0 = final[0].reward if final[0].reward is not None else 0.0
        r1 = final[1].reward if final[1].reward is not None else 0.0
        s0 = final[0].status
        s1 = final[1].status
        
        if s0 == "ERROR":
            results["A_errors"] += 1
        if s1 == "ERROR":
            results["B_errors"] += 1
            
        results["A_avg_reward"] += r0
        results["B_avg_reward"] += r1
        
        if r0 > r1:
            results["A_wins"] += 1
        elif r1 > r0:
            results["B_wins"] += 1
        else:
            results["draws"] += 1
            
        # Match 2: B is player 0, A is player 1
        env = make("crawl", configuration={"randomSeed": seed}, debug=False)
        env.run([agent_b_path, agent_a_path])
        final = env.steps[-1]
        
        r0 = final[0].reward if final[0].reward is not None else 0.0
        r1 = final[1].reward if final[1].reward is not None else 0.0
        s0 = final[0].status
        s1 = final[1].status
        
        if s0 == "ERROR":
            results["B_errors"] += 1
        if s1 == "ERROR":
            results["A_errors"] += 1
            
        results["B_avg_reward"] += r0
        results["A_avg_reward"] += r1
        
        if r0 > r1:
            results["B_wins"] += 1
        elif r1 > r0:
            results["A_wins"] += 1
        else:
            results["draws"] += 1


    total_games = num_rounds * 2
    results["A_avg_reward"] /= total_games
    results["B_avg_reward"] /= total_games
    
    print(f"=== BENCHMARK RESULTS (Total Games: {total_games}) ===")
    print(f"Agent A ({agent_a_path}): Wins: {results['A_wins']} ({results['A_wins']/total_games*100:.1f}%), Errors: {results['A_errors']}, Avg Reward: {results['A_avg_reward']:.2f}")
    print(f"Agent B ({agent_b_path}): Wins: {results['B_wins']} ({results['B_wins']/total_games*100:.1f}%), Errors: {results['B_errors']}, Avg Reward: {results['B_avg_reward']:.2f}")
    print(f"Draws: {results['draws']} ({results['draws']/total_games*100:.1f}%)")

if __name__ == "__main__":
    agent_a = "main.py"
    agent_b = "starter_bfs.py"
    if len(sys.argv) > 1:
        agent_a = sys.argv[1]
    if len(sys.argv) > 2:
        agent_b = sys.argv[2]
    
    rounds = 10
    if len(sys.argv) > 3:
        rounds = int(sys.argv[3])
    
    run_benchmark(agent_a, agent_b, num_rounds=rounds)
