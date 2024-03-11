---
description: How to integrate W&B with Stable Baseline 3.
slug: /guides/integrations/stable-baselines-3
displayed_sidebar: default
---

# Stable Baselines 3

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\)는 PyTorch에서 강화학습 알고리즘의 신뢰할 수 있는 구현체 집합입니다. W&B의 SB3 인테그레이션은 다음을 수행합니다:

* 손실 및 에피소드 별 반환과 같은 메트릭 기록
* 에이전트가 게임을 하는 비디오 업로드
* 훈련된 모델 저장
* 모델 하이퍼파라미터 로그
* 모델 그레이디언트 히스토그램 로그

[여기에서](https://wandb.ai/wandb/sb3/runs/1jyr6z10) W&B를 사용한 SB3 트레이닝 run의 예시를 확인할 수 있습니다.

## 2줄의 코드로 SB3 실험 로깅하기

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](@site/static/images/integrations/stable_baselines_demo.gif)

## WandbCallback 인수

| 인수 | 사용법 |
| :--- | :--- |
| `verbose` | sb3 출력의 상세도 |
| `model_save_path` | 모델이 저장될 폴더의 경로, 기본값은 \`None\`이므로 모델이 로그되지 않음 |
| `model_save_freq` | 모델 저장 빈도 |
| `gradient_save_freq` | 그레이디언트 로그 빈도, 기본값은 0이므로 그레이디언트가 로그되지 않음 |

## 기본 예시

W&B의 SB3 인테그레이션은 TensorBoard에서 출력하는 로그를 사용하여 메트릭을 로깅합니다.

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # 자동으로 sb3의 tensorboard 메트릭 업로드
    monitor_gym=True,  # 자동으로 에이전트가 게임하는 비디오 업로드
    save_code=True,  # 선택사항
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # 반환과 같은 통계 기록
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
```