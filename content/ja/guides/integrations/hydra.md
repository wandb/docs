---
title: Hydra
description: W&B を Hydra と統合する方法
menu:
  default:
    identifier: hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、研究やその他の複雑なアプリケーション開発をシンプルにするオープンソースの Python フレームワークです。主な特徴は、設定ファイルやコマンドラインを通じて、階層的な設定（configuration）を動的に作成・上書きできる点です。

Hydra を使って設定管理を続けながら、W&B の強力な機能も活用できます。

## メトリクスのトラッキング

`wandb.init()` や `wandb.Run.log()` を用いて、いつも通りメトリクスをトラッキングできます。ここでは、`wandb.entity` と `wandb.project` を hydra の設定ファイル内で定義しています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):

    # wandb を hydra 設定から初期化
    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
      run.log({"loss": loss})
```

## ハイパーパラメーターのトラッキング

Hydra では [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を標準で使って設定辞書にアクセスします。`OmegaConf` の辞書は通常の辞書のサブクラスではないため、Hydra の `Config` をそのまま `wandb.Run.config` に渡すとダッシュボードで予期せぬ挙動になります。そのため、`omegaconf.DictConfig` をプリミティブな `dict` 型へ変換してから `wandb.Run.config` に渡す必要があります。

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
  # omegaconf の設定を dict へ変換して wandb へ渡す
  with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
    run.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    run.log({"loss": loss})
    model = Model(**run.config.model.configs)
```

## マルチプロセスのトラブルシュート

プロセスの起動時にハングしてしまう場合は、[既知の問題]({{< relref "/guides/models/track/log/distributed-training.md" >}}) である可能性があります。これを解決するには、wandb のマルチプロセスプロトコルを変更してください。具体的には、`wandb.init()` の設定パラメーターで指定する方法があります。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

またはシェルからグローバル環境変数として設定することも可能です。

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーター最適化

[W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) は大規模なハイパーパラメーターサーチに最適なプラットフォームで、W&B での実験に関する詳細な可視化や洞察を、最小限のコードで手軽に得られます。Sweeps は Hydra プロジェクトともシームレスに統合でき、追加のコーディングは不要です。必要なのは、通常通り sweep したいパラメータを定義した設定ファイルだけです。

例えば、シンプルな `sweep.yaml` ファイルは以下のようになります。

```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test/accuracy
parameters:
  dataset:
    values: [mnist, cifar10]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

sweep を実行するには：

``` bash
wandb sweep sweep.yaml` \
```

W&B は自動的にあなたのプロジェクト内で sweep を作成し、各マシンで sweep を実行するための `wandb agent` コマンドを返します。

### Hydra の defaults に存在しないパラメータを渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra では、デフォルトの設定ファイルに存在しないパラメーターもコマンドライン経由で `+` を付けて渡すことができます。たとえば、下記のようにして追加パラメーターを指定できます。

```bash
$ python program.py +experiment=some_experiment
```

このような `+` 設定には、[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) で行うような sweep は実行できません。この制約を回避するには、experiment パラメータを空のファイルなどで初期化し、W&B Sweep で各回の呼び出しごとにその空設定を上書きしてください。詳しくは [この W&B Report](https://wandb.me/hydra) を参照してください。