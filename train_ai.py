import multiprocessing
from flappy_env import flappy_env, get_state
from dqn_agent import DQNAgent
import numpy as np
import pygame
import tensorflow as tf

# Configuração do TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
print("Versão do TensorFlow:", tf.__version__)
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU detectada e configurada!")
    except RuntimeError as e:
        print(f"Erro ao configurar a GPU: {e}")
else:
    print("Nenhuma GPU detectada. O TensorFlow usará a CPU.")


def train_instance(instance_id):
    """
    Função que treina uma instância paralela do agente DQN, com renderização.
    """
    reset_game, step, render = flappy_env()

    # Configurações para cada instância
    pygame.init()
    pygame.display.set_caption(f"Flappy Bird - Instance {instance_id}")
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 600
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    state_size = 5  # bird.y, bird.velocity, pipe.x - bird.x, pipe.top, pipe.bottom
    action_size = 2  # 0: não fazer nada, 1: flap
    agent = DQNAgent(state_size, action_size)
    episodes = 500  # Número de episódios para cada instância
    batch_size = 128  # Batch maior para atualizações mais robustas

    for e in range(episodes):
        bird, pipes, game_score = reset_game()
        state = np.reshape(get_state(bird, pipes), [1, state_size])
        total_reward = 0

        while True:
            # Ação do agente
            action = agent.act(state)
            next_state, reward, done, game_score = step(bird, pipes, action, game_score)
            next_state = np.reshape(next_state, [1, state_size])

            # Memoriza a experiência
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Renderiza o jogo
            render(bird, pipes, game_score, done)

            # Se o jogo terminou
            if done:
                print(f"[Instance {instance_id}] Episode {e + 1}/{episodes} - Game Score: {game_score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break

        # Atualiza a rede neural
        agent.replay(batch_size)

    # Salva o modelo treinado para esta instância
    agent.model.save(f"flappy_bird_dqn_instance_{instance_id}.h5")


if __name__ == "__main__":
    # Número de processos paralelos
    num_processes = 4  # Aumente conforme necessário
    processes = []

    # Cria os processos para treinamento paralelo
    for i in range(num_processes):
        p = multiprocessing.Process(target=train_instance, args=(i,))
        processes.append(p)
        p.start()

    # Aguarda todos os processos terminarem
    for p in processes:
        p.join()
