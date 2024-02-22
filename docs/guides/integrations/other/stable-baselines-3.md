---
description: How to integrate W&B with Stable Baseline 3.
slug: /guides/integrations/stable-baselines-3
displayed_sidebar: default
---

# Stable Baselines 3

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\)는 PyTorch에서 강화 학습 알고리즘의 신뢰할 수 있는 구현체 세트입니다. W&B의 SB3 통합은 다음을 수행합니다:

* 손실과 에피소드 반환과 같은 메트릭 기록
* 에이전트가 게임을 플레이하는 비디오 업로드
* 학습된 모델 저장
* 모델의 하이퍼파라미터 기록
* 모델 그레이디언트 히스토그램 기록

[W&B에서의 SB3 학습 실행 예시](https://wandb.ai/wandb/sb3/runs/1jyr6z10)입니다.

## 2줄의 코드로 SB3 실험 기록하기

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](@site/static/images/integrations/stable_baselines_demo.gif)

## WandbCallback 인수

| 인수 | 사용법 |
| :--- | :--- |
| `verbose` | sb3 출력의 상세 수준 |
| `model_save_path` | 모델이 저장될 폴더 경로, 기본값은 \`None\`으로 모델이 기록되지 않음 |
| `model_save_freq` | 모델 저장 빈도 |
| `gradient_save_freq` | 그레이디언트 기록 빈도, 기본값은 0으로 그레이디언트가 기록되지 않음 |

## 기본 예제

W&B SB3 통합은 TensorBoard의 로그 출력을 사용하여 메트릭을 기록합니다.

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
    monitor_gym=True,  # 자동으로 에이전트가 게임을 플레이하는 비디오 업로드
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