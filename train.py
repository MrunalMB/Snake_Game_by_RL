from snake_game_ai import SnakeGameAI
from agent import Agent
from helper import plot

agent = Agent()
game = SnakeGameAI()
record = 0

#attributes for graph plot
scores = []
mean_scores = []
total_score = 0
record = 0

while True:
    # 1. get old state
    state_old = agent.get_state(game)

    # 2. get move
    action = agent.get_action(state_old)

    # 3. perform move and get new state
    reward, done, score = game.play_step(action)
    state_new = agent.get_state(game)

    # 4. train short memory (immediate learning)
    agent.train_short_memory(state_old, action, reward, state_new, done)

    # 5. store memory
    agent.remember(state_old, action, reward, state_new, done)

    if done:
        # 6. train long memory (learning from past)
        game.reset()
        agent.n_games += 1
        agent.train_long_memory()

         # update record
        if score > record:
            record = score

        print("Game", agent.n_games, "Score", score, "Record:", record)

        #print("Game", agent.n_games, "Score", score)

        #Plot the graph after game completion 
        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        plot(scores, mean_scores)
