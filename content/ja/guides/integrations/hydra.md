---
title: Hydra
description: W&B を Hydra と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、研究やその他の複雑なアプリケーションの開発を簡素化するオープンソースの Python フレームワークです。重要な機能は、構成を合成して階層的に動的に作成し、それを構成ファイルやコマンドラインを介してオーバーライドする能力です。

W&B の機能を活用しながら、Hydra を使った設定管理を引き続き利用できます。

## メトリクスの追跡

通常通り、`wandb.init` と `wandb.log` を用いてメトリクスを追跡します。ここでは、`wandb.entity` と `wandb.project` は hydra 設定ファイル内で定義されています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメーターの追跡

Hydra は設定辞書を操作するためのデフォルト手段として [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を利用しています。`OmegaConf` の辞書は基本的な辞書のサブクラスではないため、Hydra の `Config` を直接 `wandb.config` に渡すとダッシュボードで予期せぬ結果を引き起こします。`omegaconf.DictConfig` を基本的な `dict` 型に変換してから `wandb.config` に渡す必要があります。

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
    model = Model(**wandb.config.model.configs)
```

## マルチプロセッシングのトラブルシューティング

プロセスの開始時にハングアップする場合、[この既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})による可能性があります。これを解決するには、次のように `wandb.init` に追加の設定パラメータを追加することで、wandb のマルチプロセッシングプロトコルを変更してみてください。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバルな環境変数を設定することで:

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーターの最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は高度にスケーラブルなハイパーパラメーター探索プラットフォームで、最低限のコードスペースで W&B 実験に関する興味深い洞察と可視化を提供します。 Sweeps は Hydra プロジェクトにノーコーディングでシームレスに統合されます。必要なのは、通常のようにスイープの対象となる様々なパラメータを説明する設定ファイルです。

単純な例としての `sweep.yaml` ファイルは以下の通りです:

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

スイープを呼び出します:

```bash
wandb sweep sweep.yaml` \
```

W&B は自動的にプロジェクト内にスイープを作成し、各マシンでスイープを実行するための `wandb agent` コマンドを返します。

### Hydra デフォルトに存在しないパラメーターを渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra はデフォルトの設定ファイルに存在しない追加のパラメーターをコマンドラインを通して渡すことをサポートしており、コマンド前に `+` を付けることで可能です。例えば、一部の値とともに追加のパラメーターを渡すには、以下のように単に呼び出します:

```bash
$ python program.py +experiment=some_experiment
```

このような `+` 設定に対して、[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) の設定時と同様にスイープすることはできません。この問題を回避するために、実験パラメーターをデフォルトの空ファイルで初期化し、W&B Sweep を用いて各呼び出し時にこれらの空の設定をオーバーライドすることができます。詳細については、[**この W&B Report**](http://wandb.me/hydra) をご覧ください。