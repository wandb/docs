---
title: Stable Baselines 3
description: W&B を Stable Baseline 3 と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) (SB3) は、PyTorch で実装された信頼性の高い強化学習アルゴリズムのセットです。W&B の SB3 インテグレーション:

* 損失やエピソードリターンなどの メトリクス を記録します。
* エージェント がゲームをプレイする動画をアップロードします。
* トレーニング された モデル を保存します。
* モデル の ハイパーパラメータ を ログ します。
* モデル の 勾配 ヒストグラム を ログ します。

W&B での SB3 トレーニング run の[例](https://wandb.ai/wandb/sb3/runs/1jyr6z10)を確認してください。

## SB3 の 実験 を ログ に記録する

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="" >}}

## WandbCallback の 引数

| 引数 | 使用法 |
| :--- | :--- |
| `verbose` | sb3 の出力の冗長性 |
| `model_save_path` | モデル が保存されるフォルダーへのパス。デフォルト値は \`None\` なので、モデル は ログ に記録されません |
| `model_save_freq` | モデル を保存する頻度 |
| `gradient_save_freq` | 勾配 を ログ に記録する頻度。デフォルト値は 0 なので、 勾配 は ログ に記録されません |

## 基本的な例

W&B SB3 インテグレーション は、TensorBoard からの ログ 出力を使用して、 メトリクス を ログ に記録します。

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
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
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