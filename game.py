import pygame
import config
from base import Base
from bird import Bird
from pipe import Pipe

pygame.init()
font = pygame.font.SysFont('lato', 30)

class Game:
    def __init__(self, w=570, h=800):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, n_games=0, best_score = 0):
        self.best_score = best_score
        self.n_games = n_games + 1

        self.bird = Bird(230, 350)
        self.pipes = [Pipe(600)]
        self.base = Base(730)

        self.score = 0

    def _update_ui(self, gen=1):
        self.display.blit(config.BG_IMG, (0,0))

        text1 = font.render("Score: " + str(self.score), True, config.WHITE)
        text2 = font.render("Generation: " + str(gen), True, config.WHITE)
        text3 = font.render("High Score: " + str(self.best_score), True, config.WHITE)

        self.display.blit(text1, [10, 5])
        self.display.blit(text2, [10, 25])
        self.display.blit(text3, [360, 5])

        for pipe in self.pipes:
            pipe.draw(self.display)

        self.base.draw(self.display)
        self.bird.draw(self.display)
        pygame.display.update()

    def play_step(self):
        # 1. collect user inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.bird.jump()

        # 2. move bird
        self.bird.move()

        # 3. check collision and move pipes
        add_pipe = False
        game_over = False
        rem = [] # pipes to remove
        for pipe in self.pipes:
            if pipe.collide(self.bird):
                # game_over = True
                return game_over, self.score

            if pipe.x + pipe.PIPE_TOP.get_width() < -20:
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1
                add_pipe = True

            pipe.move()
        
        # 4. add pipe if passed
        if add_pipe:
            self.pipes.append(Pipe(530))
        
        pipe_idx = 0
        pipe_width = self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width()
        if len(self.pipes)>1 and self.bird.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1

        # if self.bird.x > pipe_width and pipe_width < 20:
        #     pipe_vertical_distance = self.pipes[pipe_idx].bottom - self.pipes[pipe_idx+1].bottom
        #     print(pipe_vertical_distance)

        # 5. remove passed pipes
        for r in rem:
            self.pipes.remove(r)

        self.base.move()

        # 6. check if bird falls beyond base
        bird_height = self.bird.y + self.bird.img.get_height()
        if bird_height >= self.base.y or bird_height <= 0:
            game_over = True
            return game_over, self.score

        # 7. update ui
        self._update_ui(self.n_games)
        self.clock.tick(config.SPEED)

        # 8. return game over and score
        return game_over, self.score

def run():
    game = Game()
    high_score = 0
    n_games = 0

    while True:
        game_over, score = game.play_step()

        if score > high_score:
            high_score = score

        if game_over:
            game.reset(n_games, high_score)
            n_games += 1


if __name__ == '__main__':
    run()