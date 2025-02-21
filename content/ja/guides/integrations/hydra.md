---
title: Hydra
description: Hydra と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、研究やその他の複雑なアプリケーションの開発を簡素化するオープンソースの Python フレームワークです。主な機能は、構成を構成によって動的に階層的に作成し、設定ファイルやコマンドラインを通じてそれをオーバーライドできる能力です。

設定管理に Hydra を引き続き使用しながら、W&B の強力な機能を活用できます。

## メトリクスをトラックする

`wandb.init` と `wandb.log` で通常通りにメトリクスをトラックします。ここで、`wandb.entity` と `wandb.project` は Hydra の設定ファイル内で定義されています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメータをトラックする

Hydra は [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) をデフォルトの設定辞書とインターフェースする方法として使用しています。`OmegaConf` の辞書はプリミティブ辞書のサブクラスではないので、Hydra の `Config` を `wandb.config` に直接渡すと、ダッシュボードで予期しない結果が生じます。`omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.config` に渡す必要があります。

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

## マルチプロセッシングのトラブルシュート

プロセスの開始時にハングする場合は、[この既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})が原因の可能性があります。これを解決するには、wandb のマルチプロセッシングプロトコルを変更してみてください。以下のように \`wandb.init\` に追加の設定パラメータを追加します:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバル環境変数を設定します:

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメータの最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は非常にスケーラブルなハイパーパラメータ検索プラットフォームで、最小限のコードで W&B の実験に関する興味深い洞察と可視化を提供します。Sweeps はコーディングの要件なしに Hydra プロジェクトとシームレスに統合されます。必要なのは、スイープするさまざまなパラメータを通常通りに記述する設定ファイルだけです。

簡単な `sweep.yaml` ファイルの例は次のようになります:

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

スイープを開始します:

``` bash
wandb sweep sweep.yaml \
```

W&B はプロジェクト内に自動的にスイープを作成し、各マシンでスイープを実行するための `wandb agent` コマンドを提供します。

### Hydra のデフォルトに存在しないパラメータを渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra は、デフォルトの設定ファイルに存在しない追加パラメータを `+` をコマンドの前に付けることでコマンドラインから渡すことをサポートしています。たとえば、次のようにして、何らかの値で追加パラメータを渡すことができます:

```bash
$ python program.py +experiment=some_experiment
```

このような `+` 構成を Hydra 実験を設定する際と同様にスイープすることはできません。これを回避するには、実験パラメータをデフォルトの空ファイルで初期化し、W&B Sweep を使用して各コールの際にこれらの空の設定をオーバーライドします。詳細については、[**この W&B レポート**](http://wandb.me/hydra)をお読みください。