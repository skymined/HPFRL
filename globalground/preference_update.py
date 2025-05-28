# 생성된 두 개의 궤적 중 하나를 무작위로 선택하고 (사용자 요청), 선택된 궤적의 데이터를 사용하여 글로벌 액터를 업데이트

# global/preference_update.py
import torch
import torch.optim as optim
import numpy as np
import random
from local.local_actor_critic import LocalActor # Actor definition
from local.local_model import LocalDataset # Dataset definition
from torch.utils.data import DataLoader

def simulate_preference_random(trajectory1, reward1, trajectory2, reward2):
    """ Randomly selects one of the two trajectories """
    print(f"[Preference Simulation] Trajectory 1 Reward: {reward1:.2f}, Trajectory 2 Reward: {reward2:.2f}")
    choice = random.choice([1, 2])
    if choice == 1:
        print("[Preference Simulation] Randomly selected Trajectory 1.")
        return trajectory1
    else:
        print("[Preference Simulation] Randomly selected Trajectory 2.")
        return trajectory2

def update_actor_with_preference(actor, preferred_trajectory, lr=1e-4, epochs=1, batch_size=32, device='cpu'):
    """
    Updates the global actor using the preferred trajectory data (Behavior Cloning).
    """
    print("[Preference Update] Updating global actor using the preferred trajectory...")
    # Check if trajectory is valid and has data
    if preferred_trajectory is None or len(preferred_trajectory['observations']) == 0:
        print("[Preference Update] Preferred trajectory is empty or invalid. Skipping update.")
        return actor.state_dict() # Return original weights if no update

    actor.to(device).train()
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # BC loss

    # Create dataset from the preferred trajectory
    # Need to ensure keys match LocalDataset expectations
    try:
         # Ensure actions are correctly shaped before creating dataset
        pref_actions = np.array(preferred_trajectory['actions'])
        if len(pref_actions.shape) == 1:
            pref_actions = pref_actions.reshape(-1, 1)

        # Dummy rewards/next_obs/terminals if not used, but needed for Dataset structure
        dummy_rewards = preferred_trajectory.get('rewards', np.zeros(len(pref_actions)))
        dummy_next_obs = preferred_trajectory.get('next_observations', np.zeros_like(preferred_trajectory['observations']))
        dummy_terminals = preferred_trajectory.get('terminals', np.zeros(len(pref_actions)))

        pref_dataset = LocalDataset(
            preferred_trajectory['observations'],
            pref_actions,
            dummy_rewards, # Not used in BC actor update
            dummy_next_obs, # Not used in BC actor update
            dummy_terminals # Not used in BC actor update
        )
        pref_dataloader = DataLoader(pref_dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error creating DataLoader from preferred trajectory: {e}")
        print("Trajectory keys:", preferred_trajectory.keys())
        print("Observations shape:", np.array(preferred_trajectory['observations']).shape)
        print("Actions shape:", np.array(preferred_trajectory['actions']).shape)
        actor.cpu()
        return actor.state_dict() # Return original weights

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Check if dataloader is empty
        if not pref_dataloader:
             print(f"  [Preference Update] DataLoader is empty for Epoch {epoch+1}/{epochs}.")
             break

        for obs, act, _, _, _ in pref_dataloader: # Only need obs, act for BC
            obs, act = obs.to(device), act.to(device)

            predicted_actions = actor(obs)

            # Ensure shapes match for loss calculation
            if predicted_actions.shape != act.shape:
                 print(f"Shape mismatch: predicted={predicted_actions.shape}, actual={act.shape}")
                 # Attempt to fix or skip batch
                 continue

            loss = loss_fn(predicted_actions, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Check division by zero if dataloader was empty or loop didn't run
        if len(pref_dataloader) > 0:
            avg_loss = epoch_loss / len(pref_dataloader)
            print(f"  [Preference Update] Epoch {epoch+1}/{epochs}, Avg BC Loss: {avg_loss:.4f}")
        else:
            print(f"  [Preference Update] Epoch {epoch+1}/{epochs}, No batches processed.")


    actor.cpu()
    print("[Preference Update] Global actor update finished.")
    return actor.state_dict()