---
title: Stable Baselines 3
description: W&B を Stable Baseline 3 と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\) は、PyTorch で実装された信頼性の高い強化学習アルゴリズム群です。W&B の SB3 インテグレーションは次のことができます:

* 損失やエピソードごとのリターンなどのメトリクスを記録します。
* エージェントがゲームをプレイする動画をアップロードします。
* 学習済みのモデルを保存します。
* モデルのハイパーパラメーターをログします。
* モデルの勾配ヒストグラムをログします。

[例として SB3 のトレーニング run](https://wandb.ai/wandb/sb3/runs/1jyr6z10) を確認してください。

## SB3 の実験をログに記録する

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="W&B と併用した Stable Baselines 3 のトレーニング" >}}

## WandbCallback の引数

| 引数 | 使用法 |
| :--- | :--- |
| `verbose` | SB3 の出力の詳細度 |
| `model_save_path` | モデルを保存するフォルダへのパス。デフォルトの値は `None` なのでモデルはログされません |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログする頻度。デフォルトの値は 0 なので勾配はログされません |

## 基本例

W&B の SB3 インテグレーションは、TensorBoard のログ出力を使ってメトリクスをログします。

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
    sync_tensorboard=True,  # SB3 の TensorBoard メトリクスを自動アップロード
    monitor_gym=True,  # エージェントがゲームをプレイする動画を自動アップロード
    save_code=True,  # 任意
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