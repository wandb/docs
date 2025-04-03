---
title: Stable Baselines 3
description: Stable Baseline 3 と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) (SB3) は、PyTorch で記述された強化学習アルゴリズムの信頼性の高い実装のセットです。W&B の SB3 インテグレーション:

* 損失やエピソードリターンなどのメトリクスを記録します。
* エージェントがゲームをプレイする動画をアップロードします。
* トレーニング済みのモデルを保存します。
* モデルのハイパーパラメーターをログに記録します。
* モデルの勾配ヒストグラムをログに記録します。

W&B を使用した SB3 のトレーニング run の [例](https://wandb.ai/wandb/sb3/runs/1jyr6z10)を確認してください。

## SB3 の 実験管理

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="" >}}

## WandbCallback の引数

| 引数 | 使い方 |
| :--- | :--- |
| `verbose` | sb3 出力の詳細度 |
| `model_save_path` | モデルが保存されるフォルダーへのパス。デフォルト値は \`None\` なので、モデルはログに記録されません。 |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログに記録する頻度。デフォルト値は 0 なので、勾配はログに記録されません。 |

## 基本的な例

W&B SB3 インテグレーションは、TensorBoard からのログ出力を使用してメトリクスを記録します。

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
    monitor_gym=True,  # エージェントがゲームをプレイする動画を自動アップロード
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