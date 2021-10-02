import pygame
import config
import numpy as np
from base import Base
from bird import Bird
from pipe import Pipe

pygame.init()
font = pygame.font.SysFont('lato', 30)

class GameAI:
    def __init__(self, w=500, h=800):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Flappy Bird AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, n_games = 0, high_score = 0):
        self.high_score = high_score
        self.n_games = n_games + 1

        self.bird = Bird(230, 350)
        self.pipes = [Pipe(600)]
        self.base = Base(730)

        self.score = 0

    def _update_ui(self, gen = 1):
        self.display.blit(config.BG_IMG, (0,0))

        text1 = font.render("Score: " + str(self.score), True, config.WHITE)
        text2 = font.render("Generation: " + str(gen), True, config.WHITE)
        text3 = font.render("High Score: " + str(self.high_score), True, config.WHITE)

        self.display.blit(text1, [10, 5])
        self.display.blit(text2, [10, 25])
        self.display.blit(text3, [360, 5])

        for pipe in self.pipes:
            pipe.draw(self.display)
        
        self.base.draw(self.display)
        self.bird.draw(self.display)
        pygame.display.update()

    def play_step(self, action):
        # 1. collect user inputs
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move bird
        self.bird.move()
        reward = 2

        if np.array_equal(action, [0, 1]): # jump
            self.bird.jump()

        # 3. check collision and move pipes
        add_pipe = False
        game_over = False
        rem = [] # store pipes to remove
        for pipe in self.pipes:
            if pipe.collide(self.bird):
                game_over = True
                reward = -10
                return game_over, self.score, reward
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:
                self.score += 1
                reward = 20
                pipe.passed = True
                add_pipe = True
            
            pipe.move()

        # 3.1 check if bird y pos between gap coordinates
        pipe_idx = 0
        if len(self.pipes) > 1 and self.bird.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1
        
        if self.bird.y > self.pipes[pipe_idx].height and self.bird.y < self.pipes[pipe_idx].bottom:
            reward = 5

        # 4. add pipe if passed
        if add_pipe:
            self.pipes.append(Pipe(600))

        # 5. remove pipes once beyond screen
        for r in rem:
            self.pipes.remove(r)
        
        self.base.move()

        # 6. check bird falls beyond base
        bird_height = self.bird.y + self.bird.img.get_height()
        if bird_height >= self.base.y or bird_height <= 0:
            reward = -10
            game_over = True
            return game_over, self.score, reward
        
        # 7. update ui
        self._update_ui(self.n_games)
        self.clock.tick(config.SPEED)

        # 8. return gameover, score and reward
        return game_over, self.score, reward