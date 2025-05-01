# 역할 : 클라이언트별 로컬 데이ㅐ터를 사용해서 보상 예측 등의 모델을 학습. 여기서 학습된 가중치(state_dict)만 만환하고 모델 객체 자체는 저장하지 않음
# LocalDataset: 모델, 액터, 크리틱 학습에 공통으로 사용될 데이터셋 클래스. 크리틱 학습을 위해 다음 상태(next_observations)와 종료 여부(terminals) 정보를 포함


# local/local_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 데이터셋 정의 (모델, 액터, 크리틱 공용) ---
class LocalDataset(Dataset):
    def __init__(self, observations, actions, rewards, next_observations, terminals):
        """
        로컬 데이터로부터 PyTorch Dataset 객체를 생성합니다.
        액터-크리틱 학습을 위해 next_observations와 terminals 정보도 포함합니다.
        """
        self.observations = torch.tensor(observations, dtype=torch.float32)

        # 행동(actions) 데이터 타입을 float32로 통일하고, 2D 텐서 (N, action_dim) 형태로 변환합니다.
        if actions.dtype == np.float64:
            actions = actions.astype(np.float32)
        if len(actions.shape) == 1: # 만약 (N,) 형태라면 (N, 1) 형태로 변경
            actions = actions.reshape(-1, 1)
        self.actions = torch.tensor(actions, dtype=torch.float32)

        # 보상(rewards)과 종료여부(terminals)도 2D 텐서 (N, 1) 형태로 변환합니다.
        self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        self.next_observations = torch.tensor(next_observations, dtype=torch.float32)
        self.terminals = torch.tensor(terminals, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # 데이터셋의 총 샘플 수를 반환합니다.
        return len(self.observations)

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 샘플 데이터를 반환합니다.
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.terminals[idx],
        )

# --- 로컬 모델 구조 정의 (예: 보상 예측 모델) ---
class LocalRewardModel(nn.Module):
    """ 상태(state)와 행동(action)을 입력받아 보상(reward)을 예측하는 간단한 MLP 모델 예시 """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 입력층: 상태 차원 + 행동 차원 -> 은닉층 차원
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # 은닉층
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 출력층: 은닉층 차원 -> 보상 (스칼라 값 1개)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # 상태와 행동 텐서를 연결(concatenate)하여 입력으로 사용
        x = torch.cat([state, action], dim=-1)
        # 활성화 함수 (ReLU) 적용
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 최종 보상 예측값 출력
        reward = self.fc3(x)
        return reward

# --- 로컬 모델 학습 함수 (가중치만 반환) ---
def train_local_model(model_structure, dataset, state_dim, action_dim, epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    """
    주어진 모델 구조(클래스)를 사용하여 로컬 데이터셋으로 모델을 학습시키고,
    학습된 모델의 가중치(state_dict)를 반환합니다. (모델 객체 자체는 반환하지 않음)
    """
    # 학습을 위해 모델 인스턴스를 내부적으로 생성하고 지정된 디바이스로 이동
    model = model_structure(state_dim, action_dim).to(device)
    model.train() # 학습 모드로 설정
    # 옵티마이저(Adam)와 데이터로더 설정
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 손실 함수 (평균 제곱 오차) 설정
    loss_fn = nn.MSELoss()

    print(f"  [로컬 모델 학습] 시작...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # 데이터로더에서 (obs, act, rew, _, _) 형태로 데이터를 받아옴 (보상 예측 모델은 rew만 필요)
        for batch_idx, (obs, act, rew, _, _) in enumerate(dataloader):
            # 데이터를 지정된 디바이스로 이동
            obs, act, rew = obs.to(device), act.to(device), rew.to(device)

            # 모델을 통해 보상 예측
            predicted_reward = model(obs, act)
            # 예측값과 실제 보상값(rew) 사이의 손실 계산
            loss = loss_fn(predicted_reward, rew)

            # 그래디언트 초기화, 역전파, 파라미터 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() # 배치 손실 누적

        # 에폭 평균 손실 계산 및 출력
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"  [로컬 모델 학습] 에폭 {epoch+1}/{epochs}, 평균 손실: {avg_loss:.4f}")

    model.cpu() # 가중치 반환 전 모델을 CPU 메모리로 이동 (GPU 메모리 절약)
    print(f"  [로컬 모델 학습] 완료.")
    # 학습된 모델의 가중치(state_dict) 반환
    return model.state_dict()