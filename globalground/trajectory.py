# 주어진 글로벌 액터(actor) 네트워크를 사용하여 Gym 환경과 상호작용하고, 상태, 행동, 보상 등의 시퀀스인 궤적(trajectory)을 생성


# global/trajectory.py
import gymnasium as gym
import torch
import numpy as np
# 액터 클래스 정의를 가져와야 액터 인스턴스를 사용할 수 있습니다.
from local.local_actor_critic import LocalActor

def generate_trajectory(actor, env_name, max_steps=1000, device='cpu'):
    """
    주어진 액터(actor) 네트워크를 사용하여 환경(env_name)에서 궤적(trajectory)을 생성합니다.
    actor: 사용할 액터 네트워크 인스턴스 (예: 글로벌 액터)
    env_name: Gym 환경 이름 (예: 'InvertedDoublePendulum-v4')
    max_steps: 궤적의 최대 길이
    device: 액터 연산에 사용할 디바이스
    반환값: (trajectory 딕셔너리, 총 보상) 또는 (None, 0) (실패 시)
    """
    print(f"[궤적 생성] 현재 글로벌 액터를 사용하여 {env_name} 환경에서 궤적 생성 중...")
    try:
        # Gym 환경 생성 시도
        # render_mode='human' 추가 시 시각적으로 확인 가능 (단, 서버 환경 등에서는 오류 발생 가능)
        env = gym.make(env_name)
    except Exception as e:
        print(f"오류: 환경 '{env_name}' 생성 실패. {e}")
        # 환경 생성 실패 시 None과 0 반환
        return None, 0

    # 환경 초기화, 초기 상태 얻기
    state, _ = env.reset()
    # 액터를 평가 모드로 설정하고 지정된 디바이스로 이동
    actor.eval()
    actor.to(device)

    # 궤적 데이터를 저장할 딕셔너리 초기화
    trajectory = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': [], 'timeouts': []}
    total_reward = 0.0 # 총 보상 초기화
    steps_taken = 0    # 실제 스텝 수

    # 최대 스텝 수만큼 반복
    for step in range(max_steps):
        # 액터로부터 현재 상태(state)에 대한 행동(action) 얻기
        action = actor.get_action(state, device=device)

        # 행동을 환경에 맞는 형태(numpy array)로 변환 (필요 시)
        # InvertedDoublePendulum과 같은 연속 행동 공간 환경은 보통 numpy 배열을 기대함
        action_np = np.array(action).astype(np.float32)

        try:
            # 환경에서 한 스텝 진행
            next_state, reward, terminated, truncated, info = env.step(action_np)
            steps_taken += 1
        except Exception as e:
            print(f"오류: env.step 중 오류 발생: {e}")
            print(f" - 현재 상태: {state}")
            print(f" - 선택된 행동: {action_np}")
            break # 오류 발생 시 궤적 생성 중단

        # 에피소드 종료 조건 확인 (terminated: 목표 달성 또는 실패, truncated: 최대 스텝 도달)
        done = terminated or truncated

        # 궤적 데이터 저장
        trajectory['observations'].append(state)
        # 원본 행동 데이터(numpy array) 저장
        trajectory['actions'].append(action_np)
        trajectory['rewards'].append(reward)
        trajectory['next_observations'].append(next_state)
        # 명확성을 위해 terminated와 truncated 플래그를 분리하여 저장
        trajectory['terminals'].append(terminated)
        trajectory['timeouts'].append(truncated)

        # 다음 스텝을 위해 상태 업데이트
        state = next_state
        # 총 보상 누적
        total_reward += reward

        # 에피소드가 종료되면 루프 중단
        if done:
            break

    # 환경 종료
    env.close()
    # 액터를 다시 CPU로 이동 (다른 연산에서의 메모리 사용 고려)
    actor.cpu()

    # 궤적 데이터가 유효한지 확인 (최소 1스텝이라도 진행했는지)
    if steps_taken == 0:
        print("[궤적 생성] 경고: 유효한 스텝이 생성되지 않았습니다.")
        return None, 0

    # 저장된 리스트들을 numpy 배열로 변환
    for key in trajectory:
        # 비어있는 리스트가 아닌 경우에만 변환 시도
        if trajectory[key]:
            # 행동 데이터가 다차원 배열일 수 있으므로 float32 타입 지정
            if key == 'actions':
                 trajectory[key] = np.array(trajectory[key], dtype=np.float32)
                 # 만약 스칼라 행동인데 (N,) 형태로 저장되었다면 (N, 1)로 변환 필요 시 추가
                 # if trajectory[key].ndim == 1: trajectory[key] = trajectory[key].reshape(-1, 1)
            else:
                 trajectory[key] = np.array(trajectory[key])
        else:
             # 빈 리스트는 빈 numpy 배열로 유지
            trajectory[key] = np.array([])

    # 크리틱 학습 등에 사용될 'terminals' 키를 최종 종료 여부 (terminated or truncated)로 다시 정의
    # (만약 위에서 done 플래그를 직접 저장했다면 이 과정 불필요)
    trajectory['terminals'] = np.logical_or(trajectory['terminals'], trajectory['timeouts']).astype(np.float32)


    print(f"[궤적 생성] 완료. 길이: {steps_taken}, 총 보상: {total_reward:.2f}")
    return trajectory, total_reward