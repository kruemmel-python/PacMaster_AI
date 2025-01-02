import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam
import os
import random
import time
from collections import deque
import math

# Konfiguration
WIDTH, HEIGHT = 400, 400
NUM_EPISODES = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_RATE = 0.0002 # Langsamere Zerfallsrate für mehr Exploration
TARGET_UPDATE_FREQ = 100
MODEL_PATH = "ddqn_agent.keras"
RENDER_EVERY = 10
PRIORITIZED_REPLAY_EPS = 1e-6
MEMORY_SIZE_START = 1000  # Startgröße des Speichers
MEMORY_SIZE_MAX = 10000 # Maximale Speichergröße
ACTIONS = ["up", "down", "left", "right"]
NUM_ACTIONS = len(ACTIONS)

# Belohnungen
REWARD_PELLET = 2 # Increased reward for pellets
REWARD_POWER_PELLET = 10 # Increased reward for power pellets
REWARD_GHOST = -10
REWARD_WALL = -0.1
REWARD_STEP = -0.005  # Reduced step penalty

STATE_SIZE = 17 # Global static state size

class PacManEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pac-Man")
        self.clock = pygame.time.Clock()
        self.reset()
        self.ghost_speed = 4  # Geschwindigkeit der Geister
        self.power_mode = False  # Ist Pacman im Power-Modus
        self.power_mode_timer = 0 # Counter für Power Mode
        self.ghost_respawn_timer = 0
        self.ghost_respawn_delay = 100 # Nach so vielen Schritten sollen die Geister wieder erscheinen

    def reset(self):
        self.pacman_pos = [WIDTH // 2, HEIGHT // 2]
        self.ghosts_pos = [[random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)] for _ in range(4)]
        self.pellets = self.generate_pellets()
        self.power_pellets = self.generate_power_pellets()
        self.done = False
        self.power_mode = False # Zu Beginn ist der Pacman nicht im Power-Modus
        self.power_mode_timer = 0
        self.ghost_respawn_timer = 0
        return self.get_state()

    def generate_pellets(self):
        return [[random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)] for _ in range(30)] # Weniger Pellets

    def generate_power_pellets(self):
        return [[random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)] for _ in range(2)] # Weniger Power Pellets

    def get_state(self):
        state = []

        # Normalisierte Pacman Position
        state.append(self.pacman_pos[0] / WIDTH)
        state.append(self.pacman_pos[1] / HEIGHT)
      
        # Normalisierte Geisterpositionen
        for ghost in self.ghosts_pos:
            state.append(ghost[0] / WIDTH)
            state.append(ghost[1] / HEIGHT)
      
        # Relativpositionen des nächsten Pellets und Power-Pellets (normalisiert)
        closest_pellet = self.get_closest(self.pacman_pos, self.pellets)
        if closest_pellet:
            state.append((closest_pellet[0] - self.pacman_pos[0]) / WIDTH)
            state.append((closest_pellet[1] - self.pacman_pos[1]) / HEIGHT)
        else:
            state.extend([0, 0])

        closest_power_pellet = self.get_closest(self.pacman_pos, self.power_pellets)
        if closest_power_pellet:
            state.append((closest_power_pellet[0] - self.pacman_pos[0]) / WIDTH)
            state.append((closest_power_pellet[1] - self.pacman_pos[1]) / HEIGHT)
        else:
            state.extend([0, 0])

        # Normalisierte Distanzen zu den Geistern (max 4)
        ghost_distances = []
        if len(self.ghosts_pos) > 0:
          for ghost in self.ghosts_pos:
            distance = np.sqrt((ghost[0] - self.pacman_pos[0])**2 + (ghost[1] - self.pacman_pos[1])**2)
            ghost_distances.append(distance / np.sqrt(WIDTH**2 + HEIGHT**2)) # Normiere auf max Distanz
            
        while len(ghost_distances) < 4: # Fill with 0 if not enough ghosts
          ghost_distances.append(0) 
        
        state.extend(ghost_distances)
        
        # Ist Pacman im Power-Modus
        state.append(int(self.power_mode)) # 1 falls ja, 0 falls nein
        
        # Pad or truncate to STATE_SIZE
        if len(state) > STATE_SIZE:
            state = state[:STATE_SIZE] # truncate to STATE_SIZE if its too long
        while len(state) < STATE_SIZE:
            state.append(0) # Fill with 0 if its too short
        
        return np.array(state, dtype=np.float32)

    def calculate_state_size(self):
       return STATE_SIZE # static

    def get_closest(self, pos, targets):
        if not targets:
            return None
        
        closest_target = None
        min_distance = float('inf')

        for target in targets:
            distance = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_target = target
        return closest_target


    def move_ghosts(self):
        # Test: Geister stehen bleiben für Testzwecke
        # return
        for i, ghost in enumerate(self.ghosts_pos):
            direction = np.array(self.pacman_pos) - np.array(ghost)
            distance = np.linalg.norm(direction)
            
            # Bewege dich nur wenn Distanz gross genug
            if distance > 2 :
                direction = direction / distance  # Normalisiere den Richtungsvektor
                new_ghost_pos = np.array(ghost) + direction * self.ghost_speed
                self.ghosts_pos[i] = [int(new_ghost_pos[0]), int(new_ghost_pos[1])]
            
            # Stelle sicher, dass die Geister im Bildschirm bleiben
            self.ghosts_pos[i][0] = max(0, min(self.ghosts_pos[i][0], WIDTH))
            self.ghosts_pos[i][1] = max(0, min(self.ghosts_pos[i][1], HEIGHT))

    def respawn_ghosts(self):
        self.ghosts_pos = [[random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)] for _ in range(4)]


    def step(self, action):
        original_pacman_pos = list(self.pacman_pos)
        reward = REWARD_STEP # Strafe für unnötige Schritte
        
        if self.power_mode:
          self.power_mode_timer -= 1
          if self.power_mode_timer <= 0:
            self.power_mode = False # Power-Mode ist beendet

        if action == "up":
            self.pacman_pos[1] -= 10
        elif action == "down":
            self.pacman_pos[1] += 10
        elif action == "left":
            self.pacman_pos[0] -= 10
        elif action == "right":
            self.pacman_pos[0] += 10
            
        # Überprüfe auf Randüberschreitung und korrigiere die Position
        self.pacman_pos[0] = max(0, min(self.pacman_pos[0], WIDTH))
        self.pacman_pos[1] = max(0, min(self.pacman_pos[1], HEIGHT))
            
        # Setze die Position wieder zurück, wenn Pac-Man aus der Karte gelaufen wäre
        if self.pacman_pos != original_pacman_pos:
            if self.pacman_pos[0] < 0 or self.pacman_pos[0] > WIDTH or self.pacman_pos[1] < 0 or self.pacman_pos[1] > HEIGHT:
                 self.pacman_pos = original_pacman_pos
                 reward += REWARD_WALL

        # Bewege die Geister
        self.move_ghosts()
        
        # Ghost Respawn Logik
        if len(self.ghosts_pos) == 0:
            self.ghost_respawn_timer +=1
            if self.ghost_respawn_timer >= self.ghost_respawn_delay:
              self.respawn_ghosts()
              self.ghost_respawn_timer = 0


        # Kollision mit Geistern
        for ghost in self.ghosts_pos:
            if self.is_collision(self.pacman_pos, ghost):
               if not self.power_mode:
                  reward = REWARD_GHOST
                  self.done = True
               else:
                   self.ghosts_pos.remove(ghost) # Geist wird gefressen
                   reward += REWARD_POWER_PELLET # Belohnung fürs Fressen
                   break # Nach dem Fressen nur einen Geist pro Schritt
               

        # Pellets einsammeln
        for pellet in self.pellets:
            if self.is_collision(self.pacman_pos, pellet):
                self.pellets.remove(pellet)
                reward += REWARD_PELLET

        # Power-Pellets einsammeln
        for power_pellet in self.power_pellets:
            if self.is_collision(self.pacman_pos, power_pellet):
                self.power_pellets.remove(power_pellet)
                self.power_mode = True
                self.power_mode_timer = 150 # Anzahl Schritte für Power-Modus
                reward += REWARD_POWER_PELLET

        return self.get_state(), reward, self.done

    def is_collision(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) < 10 and abs(pos1[1] - pos2[1]) < 10

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 255, 0), tuple(self.pacman_pos[:2]), 10)
        for ghost in self.ghosts_pos:
            pygame.draw.circle(self.screen, (255, 0, 0), tuple(ghost), 10)
        for pellet in self.pellets:
            pygame.draw.circle(self.screen, (255, 255, 255), tuple(pellet), 5)
        for power_pellet in self.power_pellets:
            pygame.draw.circle(self.screen, (0, 255, 0), tuple(power_pellet), 7)
        
        # Zeige an, wenn Pacman im Power-Mode ist
        if self.power_mode:
            pygame.draw.circle(self.screen, (0, 255, 255), tuple(self.pacman_pos[:2]), 12)
        
        pygame.display.flip()
        self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0


    def add(self, state, action, reward, next_state, done, error):
      max_priority = max(self.priorities, default=1) if self.buffer else 1

      self.buffer.append((state, action, reward, next_state, done))
      self.priorities.append(max_priority)

    def update_beta(self):
      self.frame +=1
      self.beta = min(1.0, self.beta_start + (self.frame * (1.0 - self.beta_start) / self.beta_frames))


    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + PRIORITIZED_REPLAY_EPS) ** self.alpha


    def sample(self, batch_size):
      if len(self.buffer) < batch_size:
        return None
      
      self.update_beta()

      priorities = np.array(self.priorities)
      probs = priorities / priorities.sum()
      
      indices = np.random.choice(len(self.buffer), batch_size, p=probs)
      samples = [self.buffer[idx] for idx in indices]

      states = np.array([sample[0] for sample in samples])
      actions = np.array([sample[1] for sample in samples])
      rewards = np.array([sample[2] for sample in samples])
      next_states = np.array([sample[3] for sample in samples])
      dones = np.array([sample[4] for sample in samples])

      weights = (len(self.buffer) * probs[indices]) ** -self.beta
      weights = weights / weights.max()

      return states, actions, rewards, next_states, dones, indices, weights
    
    def __len__(self):
        return len(self.buffer)

@tf.keras.utils.register_keras_serializable()
def custom_output(tensor):
    value, advantages = tf.split(tensor, num_or_size_splits=[1, NUM_ACTIONS], axis=-1)
    return value + (advantages - tf.reduce_mean(advantages, axis=-1, keepdims=True))


class DDQNAgent:
    def __init__(self, state_size, action_size, alpha=0.6, beta_start=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(max_size=MEMORY_SIZE_MAX, alpha=alpha, beta_start=beta_start)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_rate = EPSILON_DECAY_RATE
        self.target_update_freq = TARGET_UPDATE_FREQ
        self.model = self.load_or_build_model()
        self.target_model = self.load_or_build_model()
        self.update_target_model()
        self.training_step = 0

    def load_or_build_model(self):
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
        else:
            return self.build_model()
    
    def build_model(self):
        input_layer = Input(shape=(self.state_size,))
        
        # Shared layers with Layer Normalization
        shared_dense_1 = Dense(128, activation='relu')(input_layer)
        shared_norm_1 = LayerNormalization()(shared_dense_1)
        
        shared_dense_2 = Dense(128, activation='relu')(shared_norm_1)
        shared_norm_2 = LayerNormalization()(shared_dense_2)

        # Value stream
        value_dense_1 = Dense(64, activation='relu')(shared_norm_2)
        value_norm_1 = LayerNormalization()(value_dense_1)
        value_output = Dense(1, activation='linear')(value_norm_1)

        # Advantage stream
        advantage_dense_1 = Dense(64, activation='relu')(shared_norm_2)
        advantage_norm_1 = LayerNormalization()(advantage_dense_1)
        advantage_output = Dense(self.action_size, activation='linear')(advantage_norm_1)
        
        # Combine streams
        output_layer = concatenate([value_output, advantage_output])
        output = Lambda(custom_output)(output_layer)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model


    def remember(self, state, action, reward, next_state, done, error):
        self.memory.add(state, action, reward, next_state, done, error)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_reshaped = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state_reshaped, verbose=0)

        return np.argmax(act_values[0])

    def replay(self, batch_size):
      
      sampled_data = self.memory.sample(batch_size)

      if sampled_data is None:
         return
      
      states, actions, rewards, next_states, dones, indices, weights = sampled_data

      # Predict Q-values for current states and next states using batches
      q_values = self.model.predict(states, verbose=0)
      next_q_values = self.target_model.predict(next_states, verbose=0)

      errors = []
      targets = np.zeros_like(q_values) # Initialize targets

      for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
          if done:
              target_q = reward # If done, target is simply the reward
          else:
              target_q = reward + self.gamma * np.max(next_q_values[i]) # If not done, use next_q_value
            
          errors.append(abs(q_values[i][action] - target_q)) # Calculate error
          targets[i][action] = target_q  # Store the target

      loss = self.model.train_on_batch(states, targets, sample_weight=weights)
      self.memory.update_priorities(indices, errors)

      
      # Epsilon Decay Annealing
      self.epsilon = max(self.epsilon_min, EPSILON_START - self.training_step * self.epsilon_decay_rate)
      self.training_step +=1


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def save_model(self):
        self.model.save(MODEL_PATH)

def test_environment():
    env = PacManEnvironment()
    env.reset()
    
    # Geister bewegen sich nicht und weniger Pellets
    for _ in range(200):
        env.handle_events()
        env.render()
        action = random.choice(ACTIONS) # Zufällige Aktionen
        _, _, done = env.step(action)
        if done:
            break
        time.sleep(0.05)

    print("Environment Test abgeschlossen.")

def train_agent(alpha=0.6, beta_start=0.4):
    env = PacManEnvironment()
    state_size = env.calculate_state_size()
    action_size = NUM_ACTIONS
    agent = DDQNAgent(state_size, action_size, alpha=alpha, beta_start=beta_start)
    episode_rewards = []

    for e in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(ACTIONS[action])

            # Calculate error for the prioritized replay buffer
            target = reward
            if not done:
              if next_state is not None:
                   # We predict the next state q-values, but not inside the loop
                  target = reward + GAMMA * np.max(agent.target_model.predict(np.reshape(next_state, [1, agent.state_size]), verbose=0)) # Use stored value
            
            # We predict the current state q-values, but not inside the loop
            current_q_value = agent.model.predict(np.reshape(state, [1, agent.state_size]), verbose=0)[0][action]
            error = abs(current_q_value - target) 

            agent.remember(state, action, reward, next_state, done, error)
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % RENDER_EVERY == 0:
               env.handle_events()
               env.render()

        end_time = time.time()
        episode_duration = end_time - start_time

        episode_rewards.append(total_reward)
        print(
            f"Episode: {e + 1}/{NUM_EPISODES}, Belohnung: {total_reward:.2f}, Schritte: {steps}, Epsilon: {agent.epsilon:.2f}, Dauer: {episode_duration:.2f}s, Memory size: {len(agent.memory)}"
        )

        if len(agent.memory) > BATCH_SIZE: # Train nur wenn genug Daten im Memory sind
            agent.replay(BATCH_SIZE)
        
        if e % agent.target_update_freq == 0:
             agent.update_target_model()

        agent.save_model()

    print(f"Training abgeschlossen. Durchschnittliche Belohnung: {np.mean(episode_rewards):.2f}")
    return np.mean(episode_rewards)


if __name__ == "__main__":
    # 1. Umgebungsprüfung
    test_environment()

    # 2. Visualisierung: In der Testumgebung und während des Trainings
    # (siehe train_agent, wo RENDER_EVERY verwendet wird)
    
    # 3. Test verschiedener Buffer (Priorisiert vs. Normal) - Hier wird Priorized genutzt
    # Test mit unterschiedlichen Alpha und Beta Werten
    best_reward = float('-inf')
    best_params = None
    
    # Beispiel für die Suche von Hyperparametern:
    alpha_values = [0.5, 0.6, 0.7]
    beta_start_values = [0.3, 0.4, 0.5]

    for alpha in alpha_values:
      for beta_start in beta_start_values:
        print(f"\nTraining mit alpha: {alpha}, beta_start: {beta_start}")
        avg_reward = train_agent(alpha=alpha, beta_start=beta_start)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = (alpha, beta_start)
            print("BESTER REWARD BIS JETZT: ", best_reward)


    print("\nBestes Training:")
    print(f"Alpha: {best_params[0]}, Beta_start: {best_params[1]}, Durchschnittlicher Reward: {best_reward:.2f}")


    # 4. Epsilon-Decay-Rate wird bereits in der Konfiguration oben angepasst
    
    # 5. Testen: siehe das gesamte train_agent()
