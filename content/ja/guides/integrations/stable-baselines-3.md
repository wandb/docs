---
title: Stable Baselines 3
description: W&B を Stable Baseline 3 と統合する方法
menu:
  default:
    identifier: stable-baselines-3
    parent: integrations
weight: 420
---

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)（SB3）は、PyTorch で実装された信頼性の高い強化学習アルゴリズムのセットです。W&B の SB3 インテグレーションは以下のことが可能です。

* 損失やエピソードごとのリターンなどのメトリクスを記録します。
* エージェントがゲームをプレイする様子の動画をアップロードします。
* 学習済みモデルを保存します。
* モデルのハイパーパラメーターをログします。
* モデルの勾配ヒストグラムをログします。

[SB3 トレーニング run の例](https://wandb.ai/wandb/sb3/runs/1jyr6z10)をご覧ください。

## SB3 実験のログ方法

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

{{< img src="/images/integrations/stable_baselines_demo.gif" alt="Stable Baselines 3 training with W&B" >}}

## WandbCallback の引数

| 引数 | 用途 |
| :--- | :--- |
| `verbose` | sb3 の出力の詳細レベル |
| `model_save_path` | モデルを保存するフォルダのパス。デフォルトは \`None\` なのでモデルはログされません |
| `model_save_freq` | モデルを保存する頻度 |
| `gradient_save_freq` | 勾配をログする頻度。デフォルトは 0 で勾配はログされません |

## 基本的な例

W&B SB3 インテグレーションは、TensorBoard のログ出力を利用してメトリクスを記録します。

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
    sync_tensorboard=True,  # sb3 の TensorBoard メトリクスを自動アップロード
    monitor_gym=True,  # エージェントがプレイしている動画を自動アップロード
    save_code=True,  # オプション
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