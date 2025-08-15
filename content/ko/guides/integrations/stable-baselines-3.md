---
title: Stable Baselines 3
description: W&B를 Stable Baseline 3와 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) (SB3)는 PyTorch로 구현된 신뢰할 수 있는 강화 학습 알고리즘 컬렉션입니다. W&B의 SB3 인테그레이션은 다음과 같은 기능을 제공합니다:

* 손실(loss)이나 에피소드별 보상과 같은 메트릭을 기록합니다.
* 에이전트가 게임을 플레이하는 영상을 업로드합니다.
* 트레이닝된 모델을 저장합니다.
* 모델의 하이퍼파라미터를 로그로 남깁니다.
* 모델의 그레이디언트 히스토그램을 로그로 남깁니다.

[예시 SB3 트레이닝 run](https://wandb.ai/wandb/sb3/runs/1jyr6z10)을 확인해보세요.

## SB3 Experiments 로그 남기기

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="Stable Baselines 3 training with W&B" >}}

## WandbCallback 인수

| 인수 | 용도 |
| :--- | :--- |
| `verbose` | sb3 출력의 상세 레벨 |
| `model_save_path` | 모델이 저장될 폴더 경로입니다. 기본값은 \`None\`이며, 이 경우 모델은 로그되지 않습니다 |
| `model_save_freq` | 모델 저장 주기 |
| `gradient_save_freq` | 그레이디언트 로그 주기. 기본값은 0이며, 그레이디언트는 로그되지 않습니다 |

## 기본 예제

W&B SB3 인테그레이션은 TensorBoard에서 출력된 로그를 사용해 메트릭을 기록합니다.

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
    sync_tensorboard=True,  # sb3의 tensorboard 메트릭을 자동 업로드
    monitor_gym=True,  # 에이전트 플레이 영상을 자동 업로드
    save_code=True,  # 선택사항
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # 보상 등 통계를 기록
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