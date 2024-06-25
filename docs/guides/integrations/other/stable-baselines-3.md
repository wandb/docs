---
description: W&B を Stable Baseline 3 と統合する方法。
slug: /guides/integrations/stable-baselines-3
displayed_sidebar: default
---


# Stable Baselines 3

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\)は、PyTorchで実装された信頼性の高い強化学習アルゴリズムのセットです。W&BのSB3インテグレーションは以下を行います:

* 損失やエピソードのリターンなどのメトリクスを記録
* ゲームをプレイするエージェントのビデオをアップロード
* トレーニング済みのモデルを保存
* モデルのハイパーパラメーターをログ
* モデルの勾配ヒストグラムをログ

[こちらが例](https://wandb.ai/wandb/sb3/runs/1jyr6z10)です。W&BでのSB3トレーニングrun

## SB3 Experimentsを2行のコードでログ

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](@site/static/images/integrations/stable_baselines_demo.gif)

## WandbCallback引数

| 引数 | 使用方法 |
| :--- | :--- |
| `verbose` | sb3出力の詳細度 |
| `model_save_path` | モデルが保存されるフォルダのパス。デフォルトは \`None\` なのでモデルはログされません |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログする頻度。デフォルトは0なので勾配はログされません |

## 基本的な例

W&B SB3インテグレーションは、TensorBoardの出力ログを使用してメトリクスをログします

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
    sync_tensorboard=True,  # sb3のtensorboardメトリクスを自動アップロード
    monitor_gym=True,  # エージェントがゲームをプレイするビデオを自動アップロード
    save_code=True,  # 任意
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # リターンなどの統計情報を記録
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