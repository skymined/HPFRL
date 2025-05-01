# 액터(정책)와 크리틱(가치 함수) 네트워크를 정의하고, 로컬 오프라인 데이터셋을 사용하여 학습


# local/local_actor_critic.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# 이전에 정의한 LocalDataset 클래스를 재사용합니다.
from .local_model import LocalDataset

# --- 액터 네트워크 정의 ---
# LocalActor: 상태(state)를 입력받아 결정론적인 행동(action)의 평균(mean)을 출력합니다. 출력 행동 값의 범위를 제한하기 위해 tanh 활성화 함수와 max_action 스케일링을 사용
class LocalActor(nn.Module):
    """ 상태(state)를 입력받아 행동(action)의 평균(mean)을 출력하는 액터 네트워크 """
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 행동의 평균값을 출력하는 레이어
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # 행동 값의 최대 범위를 지정 (예: InvertedDoublePendulum은 보통 [-1, 1])
        self.max_action = max_action
        # 참고: BC에서는 결정론적 액터로 충분하지만, PPO 등 확률적 정책이 필요하면 분산(log_std) 출력 추가

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # tanh 활성화 함수로 출력 범위를 [-1, 1]로 제한하고, max_action으로 스케일링
        mean = self.max_action * torch.tanh(self.fc_mean(x))
        return mean # 행동의 평균값 반환

    @torch.no_grad() # 추론 시에는 그래디언트 계산을 비활성화
    def get_action(self, state, device='cpu'):
        """ 주어진 상태에서 행동을 결정하여 반환 (환경 상호작용용) """
        self.eval() # 네트워크를 평가 모드로 설정 (Dropout 등 비활성화)
        # 입력 상태를 텐서로 변환하고 배치 차원 추가, 디바이스로 이동
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        # 액터 네트워크를 통해 행동 평균값 계산
        mean = self.forward(state_tensor)
        # self.train() # 필요하다면 다시 학습 모드로 돌려놓을 수 있음 (보통은 get_action만 호출 시 불필요)
        # 배치 차원 제거 후 numpy 배열로 변환하여 반환
        return mean.squeeze(0).cpu().numpy()

# --- 크리틱 네트워크 정의 ---
# LocalCritic: 상태(state)를 입력받아 해당 상태의 가치(value) V(s)를 출력
class LocalCritic(nn.Module):
    """ 상태(state)를 입력받아 상태 가치(V(s))를 출력하는 크리틱 네트워크 """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 상태 가치(스칼라 값)를 출력하는 레이어
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value # 상태 가치 V(s) 반환

# --- 액터-크리틱 학습 함수 (오프라인 데이터셋 사용) ---
# 크리틱 학습: 현재 상태 가치 critic(s)와 TD 타겟 r + gamma * critic(s') * (1 - terminal) 사이의 MSE 손실을 최소화하도록 학습
# 액터 학습: 액터가 출력하는 행동 actor(s)와 데이터셋에 있는 실제 행동 a 사이의 MSE 손실을 최소화하도록 학습
def train_local_actor_critic(actor, critic, dataset, epochs=5, batch_size=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, device='cpu'):
    """
    오프라인 데이터셋(dataset)을 사용하여 로컬 액터와 크리틱을 학습시킵니다.
    액터는 행동 복제(Behavior Cloning) 방식으로, 크리틱은 가치 함수 회귀 방식으로 학습합니다.
    학습된 액터와 크리틱의 가중치(state_dict)를 반환합니다.
    """
    # 액터와 크리틱을 학습 모드로 설정하고 디바이스로 이동
    actor.to(device).train()
    critic.to(device).train()
    # 각 네트워크에 대한 옵티마이저(Adam) 설정
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    # 데이터셋을 위한 데이터로더 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 손실 함수 정의: 액터(BC)와 크리틱(가치 회귀) 모두 MSE 사용 가능
    actor_loss_fn = nn.MSELoss()
    critic_loss_fn = nn.MSELoss()

    print(f"  [로컬 액터-크리틱 학습] 시작...")
    for epoch in range(epochs):
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        # 데이터로더에서 배치 단위로 데이터 추출
        for batch_idx, (obs, act, rew, next_obs, term) in enumerate(dataloader):
            # 데이터를 지정된 디바이스로 이동
            obs, act, rew, next_obs, term = obs.to(device), act.to(device), rew.to(device), next_obs.to(device), term.to(device)

            # --- 크리틱 업데이트 ---
            with torch.no_grad(): # 타겟값 계산 시 그래디언트 흐름 방지
                # 다음 상태(next_obs)에 대한 가치 예측
                next_values = critic(next_obs)
                # TD 타겟 계산: reward + gamma * V(next_state) * (1 - terminal)
                # terminal이 1이면 (종료 상태) 다음 상태 가치는 0으로 간주
                target_q = rew + gamma * next_values * (1.0 - term)

            # 현재 상태(obs)에 대한 가치 예측
            current_values = critic(obs)
            # 예측된 현재 가치와 TD 타겟 간의 MSE 손실 계산
            critic_loss = critic_loss_fn(current_values, target_q)

            # 크리틱 네트워크 업데이트
            critic_optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0) # 그래디언트 클리핑 (선택적)
            critic_optimizer.step()
            epoch_critic_loss += critic_loss.item() # 배치 손실 누적

            # --- 액터 업데이트 (행동 복제) ---
            # 현재 상태(obs)에 대한 행동 예측
            predicted_actions = actor(obs)
            # 예측된 행동과 데이터셋의 실제 행동(act) 간의 MSE 손실 계산 (BC)
            # act 텐서의 shape가 (batch_size, action_dim) 형태여야 함 (Dataset에서 처리됨)
            actor_loss = actor_loss_fn(predicted_actions, act)

            # 액터 네트워크 업데이트
            actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0) # 그래디언트 클리핑 (선택적)
            actor_optimizer.step()
            epoch_actor_loss += actor_loss.item() # 배치 손실 누적

        # 에폭별 평균 손실 계산 및 출력
        avg_actor_loss = epoch_actor_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_critic_loss = epoch_critic_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"  [로컬 A-C 학습] 에폭 {epoch+1}/{epochs}, 평균 액터 손실: {avg_actor_loss:.4f}, 평균 크리틱 손실: {avg_critic_loss:.4f}")

    # 학습 완료 후 네트워크를 CPU 메모리로 이동
    actor.cpu()
    critic.cpu()
    print(f"  [로컬 액터-크리틱 학습] 완료.")
    # 학습된 액터와 크리틱의 가중치(state_dict) 반환 ------------------------------------------------------------------------------
    return actor.state_dict(), critic.state_dict()