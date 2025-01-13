import pygame
import random
import numpy as np

pygame.init()

# Configurações do jogo
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

WHITE = (255, 255, 255)
FPS = 60
clock = pygame.time.Clock()

# Carregamento de imagens
BG_IMG = pygame.image.load("assets/background-day.png")
BIRD_IMG = pygame.image.load("assets/bluebird-downflap.png")
PIPE_IMG = pygame.image.load("assets/pipe-green.png")
GAME_OVER_IMG = pygame.image.load("assets/gameover.png")

BG_IMG = pygame.transform.scale(BG_IMG, (SCREEN_WIDTH, SCREEN_HEIGHT))
BIRD_IMG = pygame.transform.scale(BIRD_IMG, (50, 35))
PIPE_IMG = pygame.transform.scale(PIPE_IMG, (80, 500))
GAME_OVER_IMG = pygame.transform.scale(GAME_OVER_IMG, (200, 100))

PIPE_FLIPPED_IMG = pygame.transform.flip(PIPE_IMG, False, True)

# Classes do jogo
class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.gravity = 0.5
        self.lift = -10
        self.velocity = 0

    def draw(self):
        SCREEN.blit(BIRD_IMG, (self.x, self.y))

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity

        max_velocity = 10
        self.velocity = min(self.velocity, max_velocity)

        if self.y <= 0:
            self.y = 0
            self.velocity = 0

        if self.y >= SCREEN_HEIGHT - BIRD_IMG.get_height():
            self.y = SCREEN_HEIGHT - BIRD_IMG.get_height()
            self.velocity = 0

    def flap(self):
        self.velocity = self.lift

class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap = 150
        self.top = random.randint(100, SCREEN_HEIGHT - 200)
        self.bottom = self.top + self.gap

    def draw(self):
        SCREEN.blit(PIPE_FLIPPED_IMG, (self.x, self.top - PIPE_IMG.get_height()))
        SCREEN.blit(PIPE_IMG, (self.x, self.bottom))

    def update(self):
        self.x -= 5

    def off_screen(self):
        return self.x < -PIPE_IMG.get_width()

# Funções do ambiente
def get_state(bird, pipes):
    if len(pipes) > 0:
        next_pipe = pipes[0]
        state = [
            bird.y / SCREEN_HEIGHT,
            bird.velocity / 10,
            (next_pipe.x - bird.x) / SCREEN_WIDTH,
            next_pipe.top / SCREEN_HEIGHT,
            next_pipe.bottom / SCREEN_HEIGHT
        ]
    else:
        state = [bird.y / SCREEN_HEIGHT, bird.velocity / 10, 1, 0, 1]
    return np.array(state, dtype=np.float32)

def reset_game():
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH)]
    game_score = 0
    return bird, pipes, game_score

def step(bird, pipes, action, game_score):
    if action == 1:
        bird.flap()
    bird.update()

    reward = 0.1
    done = False

    for pipe in pipes:
        pipe.update()

        if bird.x > pipe.x + PIPE_IMG.get_width() and not hasattr(pipe, "scored"):
            pipe.scored = True
            game_score += 1
            reward = 1

        if pipe.off_screen():
            pipes.remove(pipe)
            pipes.append(Pipe(SCREEN_WIDTH))

        if (bird.x + BIRD_IMG.get_width() > pipe.x and
            bird.x < pipe.x + PIPE_IMG.get_width() and
            (bird.y < pipe.top or bird.y + BIRD_IMG.get_height() > pipe.bottom)):
            reward = -1
            done = True

    if bird.y >= SCREEN_HEIGHT - BIRD_IMG.get_height():
        reward = -1
        done = True

    if bird.y <= 0:
        reward -= 0.1
        bird.y = 0
        bird.velocity = 0

    next_pipe = pipes[0]
    state = get_state(bird, pipes)
    return state, reward, done, game_score

def render(bird, pipes, score, done=False):
    SCREEN.blit(BG_IMG, (0, 0))
    if not done:
        bird.draw()
        for pipe in pipes:
            pipe.draw()

    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    SCREEN.blit(score_text, (10, 10))

    if done:
        SCREEN.blit(GAME_OVER_IMG, ((SCREEN_WIDTH - GAME_OVER_IMG.get_width()) // 2,
                                    (SCREEN_HEIGHT - GAME_OVER_IMG.get_height()) // 2))
        font = pygame.font.Font(None, 36)
        message = font.render("Press any key to restart", True, WHITE)
        SCREEN.blit(message, ((SCREEN_WIDTH - message.get_width()) // 2, SCREEN_HEIGHT // 2 + 50))

    pygame.display.flip()
    clock.tick(FPS)

def flappy_env():
    return reset_game, step, render
