---
slug: /guides/integrations/stable-baselines-3
description: How to integrate W&B with Stable Baseline 3.
displayed_sidebar: ja
---

# Stable Baselines 3

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)（SB3）は、PyTorchで実装された一連の信頼性の高い強化学習アルゴリズムです。W&BのSB3統合により：

* 損失やエピソードのリターンなどのメトリクスを記録する
* ゲームをプレイするエージェントの動画をアップロードする
* トレーニングされたモデルを保存する
* モデルのハイパーパラメータを記録する
* モデルの勾配ヒストグラムを記録する

[こちらが](https://wandb.ai/wandb/sb3/runs/1jyr6z10) W&BとともにSB3トレーニングが実行された例です。

## コード2行でSB3の実験をログに記録する

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](@site/static/images/integrations/stable_baselines_demo.gif)

## WandbCallback 引数

| 引数 | 使用法 |
| :--- | :--- |
| `verbose` | SB3出力の詳細度 |
| `model_save_path` | モデルが保存されるフォルダへのパス。デフォルト値は `None` なので、モデルはログに記録されません |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログに記録する頻度。デフォルト値は0なので、勾配はログに記録されません |
## 基本的な例

W&B SB3との連携は、TensorBoardからのログ出力を使用して、メトリクスを記録します。

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
    monitor_gym=True,  # ゲームをプレイするエージェントの動画を自動アップロード
    save_code=True,  # 任意
)

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # 例えばリターンのような統計を記録
    return env
```
env = DummyVecEnv([make_env])

env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

モデル = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

モデル.learn(

    total_timesteps=config["total_timesteps"],

    callback=WandbCallback(

        gradient_save_freq=100,

        model_save_path=f"models/{run.id}",

        verbose=2,

    ),

)

run.finish()

```