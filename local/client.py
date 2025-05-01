# 각 연합 학습 클라이언트를 나타냅니다. 로컬 데이터셋을 관리하고, 로컬 모델/액터/크리틱의 학습을 트리거하며, 글로벌 가중치를 받아 로컬 네트워크를 업데이트하는 역할


# local/client.py
import torch
# 필요한 클래스와 함수들을 임포트합니다.
from .local_model import LocalRewardModel, LocalDataset, train_local_model
from .local_actor_critic import LocalActor, LocalCritic, train_local_actor_critic
import numpy as np

class Client:
    """ 연합 학습 클라이언트를 나타내는 클래스 """
    def __init__(self, client_id, local_data, state_dim, action_dim, device='cpu'):
        """ 클라이언트 초기화 """
        self.client_id = client_id  # 클라이언트 고유 ID
        self.device = device        # 연산에 사용할 디바이스 (CPU or CUDA)
        self.state_dim = state_dim  # 상태 공간 차원
        self.action_dim = action_dim # 행동 공간 차원

        # 전달받은 로컬 데이터(딕셔너리 형태)에서 필요한 정보 추출
        observations = local_data['observations']
        actions = local_data['actions']
        rewards = local_data['rewards']
        next_observations = local_data['next_observations'] # 크리틱 학습에 필요
        terminals = local_data['terminals']             # 크리틱 학습에 필요

        # 추출한 데이터로 로컬 데이터셋 객체 생성
        self.local_dataset = LocalDataset(observations, actions, rewards, next_observations, terminals)

        # 로컬 액터와 크리틱 네트워크 인스턴스 생성 및 디바이스 할당
        self.local_actor = LocalActor(state_dim, action_dim).to(self.device)
        self.local_critic = LocalCritic(state_dim).to(self.device)

        # 로컬 모델은 학습 시에만 임시로 생성되므로, 여기서는 모델 구조(클래스)만 저장합니다.
        self.model_structure = LocalRewardModel

    # -------------------- local_model.py의 train_local_model 함수를 호출하여 모델 학습을 수행하고, 반환된 가중치만 얻음 #
    def train_model_get_weights(self, epochs=3, batch_size=64, lr=1e-3):
        """
        로컬 모델 학습을 수행하고, 학습된 가중치(state_dict)만 반환합니다.
        클라이언트 내부에 모델 객체를 유지하지 않습니다.
        """
        print(f"[클라이언트 {self.client_id}] 로컬 모델 학습 중...")
        # train_local_model 함수 호출 (모델 구조, 데이터셋, 파라미터 전달)
        model_weights = train_local_model(
            self.model_structure, self.local_dataset, self.state_dim, self.action_dim,
            epochs, batch_size, lr, self.device
        )
        # 학습된 가중치 반환
        return model_weights
    # local_actor_critic.py의 train_local_actor_critic 함수를 호출하여 로컬 액터와 크리틱을 학습
    def train_policy(self, epochs=5, batch_size=64, lr_actor=3e-4, lr_critic=1e-3):
        """ 로컬 액터와 크리틱을 학습시키고, 학습된 가중치를 반환합니다. """
        print(f"[클라이언트 {self.client_id}] 로컬 액터-크리틱 학습 중...")
        # train_local_actor_critic 함수 호출 (액터, 크리틱, 데이터셋, 파라미터 전달)
        actor_weights, critic_weights = train_local_actor_critic(
            self.local_actor, self.local_critic, self.local_dataset,
            epochs, batch_size, lr_actor, lr_critic, device=self.device
        )
        # 학습된 가중치로 클라이언트 내부의 액터와 크리틱 네트워크를 업데이트합니다.
        self.local_actor.load_state_dict(actor_weights)
        self.local_critic.load_state_dict(critic_weights)
        # 연합 평균(aggregation)을 위해 학습된 가중치를 반환합니다.
        return actor_weights, critic_weights

    def get_actor_weights(self):
        """ 로컬 액터의 현재 가중치를 반환합니다. """
        return self.local_actor.state_dict()

    def set_actor_weights(self, global_weights):
        """ 전달받은 글로벌 액터 가중치로 로컬 액터를 업데이트합니다. """
        self.local_actor.load_state_dict(global_weights)
        # print(f"[클라이언트 {self.client_id}] 로컬 액터가 글로벌 가중치로 업데이트됨.") # 로그 출력 (선택적)

    def get_critic_weights(self):
        """ 로컬 크리틱의 현재 가중치를 반환합니다. """
        return self.local_critic.state_dict()

    def set_critic_weights(self, global_weights):
        """ 전달받은 글로벌 크리틱 가중치로 로컬 크리틱을 업데이트합니다. """
        self.local_critic.load_state_dict(global_weights)
        # print(f"[클라이언트 {self.client_id}] 로컬 크리틱이 글로벌 가중치로 업데이트됨.") # 로그 출력 (선택적)

    # 모델 객체를 클라이언트가 직접 관리하지 않으므로, 모델 가중치 get/set 메서드는 없습니다.