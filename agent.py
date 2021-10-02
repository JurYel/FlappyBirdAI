import torch
import numpy as np
import random
from gameAI import GameAI
from collections import deque
import time
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 48
LR = 0.001

class Agent:
    def __init__(self, model):
        self.n_games = 0
        self.best_score = 0
        # self.epsilon = 0 # controls random
        # self.gamma = 0.999 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeds max memory
                                               # calls popleft() 
                                               # and remove items from left
        # self.model = Linear_QNet(3, 256, 2)
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.model.gamma)

    def get_state(self, game):
        bird = game.bird
        pipes = game.pipes

        pipe_idx = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1
        
        # Strat #1:
        # - try adding incremental rewards if 
        #   bird position is between coordinates of 
        #   pipe gaps
        # - compute bird.y position
        # - check if within gap coordinates between
        #   pipe top and pipe bottom
        # - then add incremental rewards

        # Strat #2:
        # - check if bird x position with gap x coords
        # - check if bird.x and pipe.x diff = 0
        # - add increment rewards if so

        # pipe_top_coord = pipes[pipe_idx].height
        # pipe_bot_coord = pipes[pipe_idx].bottom
        # gap = abs(pipe_bot_coord - pipe_top_coord)
        # if bird.y >= pipe_top_coord or bird.y <= pipe_bot_coord:
        #     reward += 2

        # this state_space performs better 
        state = [
            # bird position
            bird.y,

            # distance to top mask
            abs(bird.y - pipes[pipe_idx].height),
            # pipes[pipe_idx].height,

            # distance to bottom mask
            abs(bird.y - pipes[pipe_idx].bottom)
            # pipes[pipe_idx].bottom,

            # get bird x pos
            # bird.x,

            # distance to pipe
            # abs(bird.x - pipes[pipe_idx].x)
        ]

        in_gap = (bird.y > pipes[pipe_idx].height and bird.y < pipes[pipe_idx].bottom)
        above_pipe = (bird.y < pipes[pipe_idx].height - 20)
        below_pipe = (bird.y > pipes[pipe_idx].bottom + 20)
        near_pipe = (bird.x - pipes[pipe_idx].x < 10)

        state_bools = [
            # bird in gap
            in_gap,

            # bird above pipe
            above_pipe,

            # bird below pipe
            below_pipe,

            # near pipe
            near_pipe
        ]

        # Proposed State Space:
        # X - Horizontal Distance to Next Pipe
        # Y - Vertical Distance to Next Pipe
        # V - Current Velocity of the bird
        # --- the 4th dimension ---
        # Y1 - if bird passed first pipe, 
        #    - calculate the vertical distance between next two pipes
        #    - it helps the bird to take action in advance
        #      according to the height difference of
        #      two consecutive pipes.
        #    - Value only used when the bird enters the
        #    - tunnel part. It can reduce the state space.

        # x = bird.x - pipes[pipe_idx].x
        # y = bird.y - pipes[pipe_idx].bottom
        # v = pipes[pipe_idx].VEL
        
        # pipe_width = pipes[0].x + pipes[0].PIPE_TOP.get_width()
        # if bird.x > pipe_width and pipe_width < 30:
            # nx_pipe_dist = pipes[pipe_idx].bottom - pipes[pipe_]


        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state, epsilon):
        # random moves: exploration-exploitation tradeoff

        # ---------INEFFICIENT PROCESS---------- #
        # self.epsilon = 80 - self.n_games
        # final_move = [0, 0]
        # # final_move = 0
        # if random.randint(0, 200) < self.epsilon:
        #     move = random.randint(0,1)
        #     final_move[move] = 1
        #     # final_move = move
        # else:
        #     state0 = torch.tensor(state, dtype=torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        #     # f_move = prediction[move]
        #     # print(f_move)
        #     final_move[move] = 1

        #     # if f_move > 0.5:
        #     #     final_move = move
        #     # return f_move

        #     # final_move = move
        # ------------------------------------- #

        final_move = [0, 0]

        # get network prediction
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        random_action = random.random() <= epsilon
        action_idx = [torch.randint(2, torch.Size([]), dtype=torch.int)
                      if random_action
                      else torch.argmax(prediction)][0]

        final_move[action_idx] = 1

        return final_move

    
def train(net, start):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0

    agent = Agent(net)
    game = GameAI()

    # init epsilon
    epsilon = net.init_epsilon

    # epsilon annealing
    epsilon_decrements = np.linspace(net.init_epsilon, net.final_epsilon, net.num_iterations)

    t = 0

    for iteration in range(net.num_iterations):
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, epsilon)

        # perform move and get new state
        done, score, reward = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (trains agent for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember all (store in memory)
        agent.remember(state_old, final_move, reward, state_new, done)

        epsilon = epsilon_decrements[iteration]

        t += 1

        if done:
            # train long memory ( replay memory/experience replay )
            # required for agent, tremendously helps training
            game.reset(agent.n_games, best_score)
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                # agent.model.save()

            print("Game: " + str(agent.n_games),
                  "Score: " + str(score),
                  "High Score: " + str(best_score))

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            net.episode_durations.append(t)
            t = 0

            # for plotting mean scores
            # plot(plot_scores, plot_mean_scores)

            # for plotting durations
            # plot_durations(net.episode_durations)

            if iteration % 250 == 0:
                net.save()

            if iteration % 100 == 0:
                print("iteration: ", iteration,
                      "elapsed time: ", time.time() - start,
                      "epsilon: ", epsilon)

if __name__ == '__main__':
    net = Linear_QNet(3, 512, 2)
    start = time.time()
    train(net, start)
