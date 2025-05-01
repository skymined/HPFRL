# 여러 클라이언트로부터 수집한 로컬 가중치들(state_dict 리스트)을 평균 내어 글로벌 가중치를 계산


# global/aggregation.py
import torch
from collections import OrderedDict
import copy # deepcopy를 위해 추가 (선택적)

# 핵심 로직 함수. 입력된 가중치 리스트(w)의 각 키(레이어)별로 텐서들을 모아 평균을 계산
# 메모리 문제 피하기 위해서 CPU로 하는게 좋을 수 있다고 함
def average_weights(w):
    """
    클라이언트들의 가중치(state_dict) 리스트 w를 받아 평균 가중치를 계산합니다.
    w: list of state_dicts from clients
    """
    if not w: # 입력 리스트가 비어있으면 None 반환
        print("Warning: 가중치 리스트가 비어있어 평균을 계산할 수 없습니다.")
        return None

    # 모든 클라이언트가 동일한 키(레이어 이름)를 가지고 있는지 확인 (선택적이지만 권장)
    first_keys = w[0].keys()
    for i, state_dict in enumerate(w[1:]):
        if state_dict.keys() != first_keys:
            print(f"Warning: 클라이언트 {i+1}의 가중치 키가 첫 번째 클라이언트와 다릅니다. 평균 계산에 문제가 발생할 수 있습니다.")
            # 여기서 에러를 발생시키거나, 공통 키만 사용하거나, 문제가 있는 클라이언트를 제외하는 등의 처리 필요
            # 가장 간단한 방법은 일단 진행하되, 아래 key별 처리에서 오류 발생 시 확인

    # 평균 가중치를 저장할 OrderedDict 초기화 (첫 번째 클라이언트 기준으로 구조 복사)
    # w_avg = OrderedDict() # 빈 딕셔너리로 시작해도 됨
    # deepcopy를 사용하여 원본 가중치 변경 방지 (메모리 사용량 증가 가능성 있음)
    try:
        w_avg = copy.deepcopy(w[0])
    except Exception as e:
        print(f"Warning: 첫 번째 가중치 deepcopy 중 오류 발생 ({e}). 빈 딕셔너리로 시작합니다.")
        w_avg = OrderedDict()


    # 모든 키(레이어 이름)에 대해 반복
    for key in w_avg.keys(): # 첫 번째 클라이언트의 키들을 기준으로 순회
        # 해당 키에 대한 모든 클라이언트의 텐서를 리스트로 수집
        # 계산 안정성을 위해 CPU로 이동하고 float 타입으로 변환
        key_tensors = []
        valid_client_count = 0
        for state_dict in w:
            if key in state_dict: # 해당 키가 클라이언트 state_dict에 있는지 확인
                 # CPU로 이동 시키고 float으로 변환
                key_tensors.append(state_dict[key].cpu().float())
                valid_client_count += 1
            else:
                print(f"Warning: 일부 클라이언트에서 키 '{key}'를 찾을 수 없습니다. 해당 클라이언트는 이 키의 평균 계산에서 제외됩니다.")

        if valid_client_count == 0:
            print(f"Error: 키 '{key}'에 대한 유효한 텐서를 찾을 수 없습니다. 평균 계산 불가.")
            # w_avg.pop(key, None) # 해당 키를 결과에서 제거하거나 None 할당 등의 처리
            w_avg[key] = None # 또는 에러 발생
            continue

        # 수집된 텐서들을 쌓아서(stack) 평균 계산 (dim=0 기준)
        try:
            stacked_tensors = torch.stack(key_tensors, dim=0)
            w_avg[key] = torch.mean(stacked_tensors, dim=0)
        except RuntimeError as e:
            print(f"Error: 키 '{key}'의 텐서들을 stack 또는 mean 하는 중 오류 발생: {e}")
            print(f" - 텐서 Shapes: {[t.shape for t in key_tensors]}")
            # 오류 처리: 해당 키 제거, None 할당 등
            w_avg[key] = None # 또는 raise e


    return w_avg

def average_models(model_weights_list):
    """ 로컬 모델 가중치 리스트를 평균냅니다. """
    print("[글로벌 집계] 로컬 모델 가중치 평균 계산 중...")
    return average_weights(model_weights_list)

def average_actors(actor_weights_list):
    """ 로컬 액터 가중치 리스트를 평균냅니다. """
    print("[글로벌 집계] 로컬 액터 가중치 평균 계산 중...")
    return average_weights(actor_weights_list)

def average_critics(critic_weights_list):
    """ 로컬 크리틱 가중치 리스트를 평균냅니다. """
    print("[글로벌 집계] 로컬 크리틱 가중치 평균 계산 중...")
    return average_weights(critic_weights_list)