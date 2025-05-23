---
title: Stable Baselines 3
description: W&B を Stable Baseline 3 と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\) は、PyTorch による強化学習アルゴリズムの信頼性のある実装セットです。W&B の SB3 インテグレーション:

* 損失やエピソードごとのリターンなどのメトリクスを記録します。
* ゲームをプレイするエージェントのビデオをアップロードします。
* トレーニング済みモデルを保存します。
* モデルのハイパーパラメーターをログします。
* モデルの勾配ヒストグラムをログします。

W&B を用いた SB3 トレーニング run の[例](https://wandb.ai/wandb/sb3/runs/1jyr6z10)をレビューしてください。

## SB3 実験をログする

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="" >}}

## WandbCallback 引数

| 引数 | 使用法 |
| :--- | :--- |
| `verbose` | sb3 出力の詳細度 |
| `model_save_path` | モデルが保存されるフォルダーへのパス。デフォルト値は `None` で、モデルはログされません。 |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログする頻度。デフォルト値は 0 で、勾配はログされません。 |

## 基本的な例

W&B SB3 インテグレーションは、TensorBoard から出力されたログを使用してメトリクスをログします。

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
    sync_tensorboard=True,  # sb3 の tensorboard メトリクスを自動アップロード
    monitor_gym=True,  # ゲームをプレイするエージェントのビデオを自動アップロード
    save_code=True,  # オプション
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # リターンなどの統計を記録
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