# main.py
import torch
import minari
import numpy as np
import gymnasium as gym
import os
from tqdm import tqdm
import random

from local.client import Client
from local.local_model import LocalRewardModel # Import structure
from local.local_actor_critic import LocalActor, LocalCritic # Import structures
from globalground.aggregation import average_models, average_actors, average_critics
from globalground.trajectory import generate_trajectory
from globalground.preference_update import simulate_preference_random, update_actor_with_preference

# --- Configuration ---
N_CLIENTS = 5
N_ROUNDS = 20 # More rounds might be needed for preference learning
LOCAL_MODEL_EPOCHS = 3
LOCAL_POLICY_EPOCHS = 5 # Epochs for local actor-critic training
BATCH_SIZE = 64
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
LR_MODEL = 1e-3
LR_PREFERENCE_UPDATE = 1e-4 # Learning rate for global actor update based on preference
GAMMA = 0.99
ENV_NAME = 'InvertedDoublePendulum-v4'
# Find a suitable Minari dataset ID, expert or otherwise
MINARI_DATASET_ID = 'relocate-cloned-v1' # EXAMPLE ID - REPLACE with a valid InvertedDoublePendulum dataset if available
# Check available datasets: print(minari.list_remote_datasets())
# Or use a known good one like: 'door-human-v1', 'pen-human-v1' - adapt dims if needed
MAX_TRAJECTORY_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
WEIGHTS_DIR = "saved_weights" # Directory to save weights

# --- Setup ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --- Load Minari Dataset & Env Info ---
print(f"Attempting to load Minari dataset: {MINARI_DATASET_ID}")
available_local = minari.list_local_datasets()
if MINARI_DATASET_ID not in available_local:
    try:
        print(f"Dataset not found locally. Downloading {MINARI_DATASET_ID}...")
        minari.download_dataset(MINARI_DATASET_ID)
    except Exception as e:
        print(f"ERROR: Failed to download Minari dataset '{MINARI_DATASET_ID}'. {e}")
        print("Please check the dataset ID and your internet connection.")
        print("Available remote datasets:", minari.list_remote_datasets())
        exit()

try:
    dataset = minari.load_dataset(MINARI_DATASET_ID)
    # Get env info from dataset or dummy env
    env_temp = dataset.recover_environment()
    if env_temp is None:
        print("Warning: Could not recover environment from dataset. Creating dummy env.")
        env_temp = gym.make(ENV_NAME)

    STATE_DIM = env_temp.observation_space.shape[0]
    ACTION_DIM = env_temp.action_space.shape[0]
    # MAX_ACTION = float(env_temp.action_space.high[0]) # Usually 1.0 for tanh activated policies
    env_temp.close()
    print(f"Environment: {ENV_NAME}, State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")
except Exception as e:
    print(f"ERROR: Failed to load dataset or get environment info: {e}")
    print("Ensure the dataset ID is correct and compatible with the environment.")
    # Fallback dimension definition if needed, adjust based on chosen ENV/DATASET
    # STATE_DIM = 11 # Example for InvertedDoublePendulum
    # ACTION_DIM = 1 # Example for InvertedDoublePendulum
    # print(f"Warning: Using fallback dimensions. State: {STATE_DIM}, Action: {ACTION_DIM}")
    exit()


# --- Prepare Client Data ---
episodes = list(dataset.iterate_episodes())
n_episodes = len(episodes)
if n_episodes == 0:
    raise ValueError(f"Minari dataset '{MINARI_DATASET_ID}' contains no episodes.")
if n_episodes < N_CLIENTS:
    print(f"Warning: Only {n_episodes} episodes available, but {N_CLIENTS} clients requested. Using modulo assignment.")
    client_episode_indices = [[] for _ in range(N_CLIENTS)]
    for i in range(n_episodes):
        client_episode_indices[i % N_CLIENTS].append(i)
else:
    indices = np.array_split(np.arange(n_episodes), N_CLIENTS)
    client_episode_indices = [idx_list.tolist() for idx_list in indices]

client_data_list = []
print("Distributing data to clients...")
for i in range(N_CLIENTS):
    client_ep_indices = client_episode_indices[i]
    if not client_ep_indices:
        print(f"Client {i} has no episodes assigned.")
        continue

    obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []
    for ep_idx in client_ep_indices:
        try:
            episode = episodes[ep_idx]
            # Ensure data has the expected keys and format
            if not all(hasattr(episode, attr) for attr in ['observations', 'actions', 'rewards', 'terminations', 'truncations']):
                 print(f"Warning: Episode {ep_idx} missing required attributes. Skipping.")
                 continue

            # Data length check might be needed depending on dataset format
            # Example: ensure obs has one more step than actions/rewards
            if len(episode.observations) != len(episode.actions) + 1:
                 print(f"Warning: Episode {ep_idx} obs/action length mismatch ({len(episode.observations)} vs {len(episode.actions)}). Trying to align.")
                 # Simple alignment: take obs[:-1] and all actions/rewards
                 obs = episode.observations[:-1]
                 next_obs = episode.observations[1:]
                 actions = episode.actions
                 rewards = episode.rewards
                 terminals = np.logical_or(episode.terminations, episode.truncations) # Combine termination and truncation
                 # Ensure lengths match after alignment
                 min_len = min(len(obs), len(actions), len(rewards), len(next_obs), len(terminals))
                 if min_len < 1 : continue # Skip if no valid steps
                 obs = obs[:min_len]
                 actions = actions[:min_len]
                 rewards = rewards[:min_len]
                 next_obs = next_obs[:min_len]
                 terminals = terminals[:min_len]

            else:
                 # Standard case assuming lengths match as expected by Minari spec often
                 obs = episode.observations[:-1]
                 actions = episode.actions
                 rewards = episode.rewards
                 next_obs = episode.observations[1:]
                 terminals = np.logical_or(episode.terminations, episode.truncations) # Combine termination and truncation

            # Dimension checks (especially for actions)
            if actions.ndim == 1: actions = actions.reshape(-1, 1) # Ensure 2D
            if actions.shape[1] != ACTION_DIM:
                print(f"Warning: Episode {ep_idx} action dim mismatch (Expected {ACTION_DIM}, Got {actions.shape[1]}). Skipping episode.")
                continue
            if obs.shape[1] != STATE_DIM:
                print(f"Warning: Episode {ep_idx} observation dim mismatch (Expected {STATE_DIM}, Got {obs.shape[1]}). Skipping episode.")
                continue

            obs_list.append(obs)
            act_list.append(actions)
            rew_list.append(rewards)
            next_obs_list.append(next_obs)
            term_list.append(terminals)

        except Exception as e:
            print(f"Error processing episode {ep_idx} for client {i}: {e}")
            continue # Skip problematic episode

    if not obs_list: # Check if any valid data was collected
        print(f"Client {i} collected no valid data.")
        continue

    client_data = {
        'observations': np.concatenate(obs_list, axis=0),
        'actions': np.concatenate(act_list, axis=0).astype(np.float32), # Ensure float32
        'rewards': np.concatenate(rew_list, axis=0),
        'next_observations': np.concatenate(next_obs_list, axis=0),
        'terminals': np.concatenate(term_list, axis=0),
    }
    client_data_list.append(client_data)

N_VALID_CLIENTS = len(client_data_list)
if N_VALID_CLIENTS == 0:
    raise ValueError("No clients have valid data after processing.")
print(f"Prepared data for {N_VALID_CLIENTS} clients.")


# --- Initialize Clients ---
clients = [Client(client_id=i, local_data=client_data_list[i], state_dim=STATE_DIM, action_dim=ACTION_DIM, device=DEVICE) for i in range(N_VALID_CLIENTS)]

# --- Initial Local Training & Aggregation ---
print("\n--- Initial Local Training Phase ---")
all_model_weights = []
all_actor_weights = []
all_critic_weights = []

for client in tqdm(clients, desc="Initial Training"):
    model_w = client.train_model_get_weights(epochs=LOCAL_MODEL_EPOCHS, batch_size=BATCH_SIZE, lr=LR_MODEL)
    actor_w, critic_w = client.train_policy(epochs=LOCAL_POLICY_EPOCHS, batch_size=BATCH_SIZE, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)

    all_model_weights.append(model_w)
    all_actor_weights.append(actor_w)
    all_critic_weights.append(critic_w)

# Aggregate initial weights
initial_global_model_weights = average_models(all_model_weights)
initial_global_actor_weights = average_actors(all_actor_weights)
initial_global_critic_weights = average_critics(all_critic_weights)

# Save the initial aggregated model weights (as requested)
if initial_global_model_weights:
    model_weights_path = os.path.join(WEIGHTS_DIR, "initial_global_model_weights.pth")
    torch.save(initial_global_model_weights, model_weights_path)
    print(f"Initial global model weights saved to {model_weights_path}")

# Create global actor and critic instances and load initial weights
global_actor = LocalActor(STATE_DIM, ACTION_DIM).to(DEVICE)
global_critic = LocalCritic(STATE_DIM).to(DEVICE) # Critic is aggregated but not used in preference update step here

if initial_global_actor_weights:
    global_actor.load_state_dict(initial_global_actor_weights)
    print("Initial global actor created.")
else:
    print("Warning: No valid actor weights to aggregate. Global actor initialized randomly.")

if initial_global_critic_weights:
    global_critic.load_state_dict(initial_global_critic_weights)
    print("Initial global critic created.")
else:
     print("Warning: No valid critic weights to aggregate. Global critic initialized randomly.")


# --- Federated Learning with Preference Loop ---
print("\n--- Federated Learning with Preference Update Phase ---")
for round_num in range(N_ROUNDS):
    print(f"\n===== Round {round_num + 1} / {N_ROUNDS} =====")

    # Step 1 & 2: Policy Distribution & Local Training (Optional for this baseline)
    # In this setup, we use the centrally updated global_actor directly.
    # If local updates per round were desired, distribute global_actor_weights here
    # and call client.train_policy() again, then re-aggregate.

    # Step 3: Generate two trajectories using the current global actor
    print("Generating trajectories...")
    trajectory1, reward1 = generate_trajectory(global_actor, ENV_NAME, MAX_TRAJECTORY_STEPS, DEVICE)
    trajectory2, reward2 = generate_trajectory(global_actor, ENV_NAME, MAX_TRAJECTORY_STEPS, DEVICE)

    # Handle cases where trajectory generation might fail
    if trajectory1 is None or trajectory2 is None:
        print("Warning: Failed to generate one or both trajectories. Skipping preference update for this round.")
        continue

    # Step 4: Simulate preference (Random Selection)
    preferred_trajectory = simulate_preference_random(trajectory1, reward1, trajectory2, reward2)

    # Step 5: Update global actor using the preferred trajectory
    updated_actor_weights = update_actor_with_preference(
        global_actor, preferred_trajectory, lr=LR_PREFERENCE_UPDATE, epochs=1, batch_size=BATCH_SIZE, device=DEVICE
    )

    # Load the updated weights into the global actor for the next round
    global_actor.load_state_dict(updated_actor_weights)

    # Step 6: Distribute updated policy (Implicit in this centralized simulation)
    # In real FL, send updated_actor_weights (and potentially critic weights if updated) to clients.
    # For this simulation, global_actor is ready for the next round.
    # We also update the clients' internal actors to reflect the global change IF they were used in local steps.
    # Although not strictly needed if no local training happens in the loop:
    # for client in clients:
    #     client.set_actor_weights(updated_actor_weights)
        # client.set_critic_weights(global_critic.state_dict()) # Distribute critic too if needed

    print(f"===== End of Round {round_num + 1} =====")


# --- Final Evaluation ---
print("\n--- Final Evaluation ---")
# Ensure global_actor is on the correct device for final eval if needed
global_actor.to(DEVICE)
final_trajectory, final_reward = generate_trajectory(global_actor, ENV_NAME, MAX_TRAJECTORY_STEPS, DEVICE)
if final_trajectory:
    print(f"Final global actor evaluation: Total Reward = {final_reward:.2f}")
else:
    print("Final evaluation failed.")

# Save final actor/critic weights (optional)
# final_actor_path = os.path.join(WEIGHTS_DIR, "final_global_actor.pth")
# final_critic_path = os.path.join(WEIGHTS_DIR, "final_global_critic.pth")
# torch.save(global_actor.state_dict(), final_actor_path)
# torch.save(global_critic.state_dict(), final_critic_path)
# print(f"Final actor weights saved to {final_actor_path}")
# print(f"Final critic weights saved to {final_critic_path}")

print("\nFederated Offline RLHF (Actor-Critic) simulation finished.")