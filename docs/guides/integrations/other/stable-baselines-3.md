---
title: Stable Baselines 3
description: W&B를 Stable Baseline 3와 통합하는 방법.
slug: /guides/integrations/stable-baselines-3
displayed_sidebar: default
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\) 는 PyTorch를 사용한 강화학습 알고리즘의 신뢰성 있는 구현 집합입니다. W&B의 SB3 인테그레이션은 다음을 수행합니다:

* 손실 및 에피소드 반환과 같은 메트릭 기록
* 에이전트가 게임을 플레이하는 비디오 업로드
* 트레이닝된 모델 저장
* 모델의 하이퍼파라미터 로그
* 모델 그레이디언트 히스토그램 로그

[여기는 W&B와 함께한 SB3 트레이닝 run의 예제](https://wandb.ai/wandb/sb3/runs/1jyr6z10)입니다.

## 두 줄의 코드로 SB3 Experiments 로그하기

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](/images/integrations/stable_baselines_demo.gif)

## WandbCallback 인수

| 인수 | 사용법 |
| :--- | :--- |
| `verbose` | sb3 출력의 자세함 정도 |
| `model_save_path` | 모델이 저장될 폴더의 경로입니다. 기본 값은 \`None\`이며, 이 경우 모델은 로그되지 않습니다 |
| `model_save_freq` | 모델을 저장하는 빈도 |
| `gradient_save_freq` | 그레이디언트를 로그하는 빈도. 기본 값은 0이며, 이 경우 그레이디언트는 로그되지 않습니다 |

## 기본 예제

W&B의 SB3 인테그레이션은 TensorBoard의 로그 출력을 사용하여 메트릭을 로그합니다.

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
    sync_tensorboard=True,  # sb3의 tensorboard 메트릭 자동 업로드
    monitor_gym=True,  # 에이전트가 게임을 플레이하는 비디오 자동 업로드
    save_code=True,  # 선택 사항
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