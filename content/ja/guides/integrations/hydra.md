---
title: Hydra
description: W&B と Hydra の統合方法について説明します。
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、 研究 およびその他の複雑な アプリケーション の開発を簡素化する、オープンソースの Python フレームワーク です。主な機能は、構成ファイルを介して階層的な 設定 を動的に作成し、 コマンドライン からオーバーライドできることです。

W&B の機能を活用しながら、引き続き Hydra を 設定 管理に使用できます。

## メトリクス の追跡

`wandb.init` と `wandb.log` を使用して、通常どおり メトリクス を追跡します。ここでは、`wandb.entity` と `wandb.project` は hydra 設定 ファイル内で定義されています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメータ の追跡

Hydra は、 設定 辞書 とのインターフェースをとるためのデフォルトの方法として [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を使用します。`OmegaConf` の 辞書 は、プリミティブな 辞書 のサブクラスではないため、Hydra の `Config` を `wandb.config` に直接渡すと、 ダッシュボード で予期しない 結果 が発生します。`omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.config` に渡す必要があります。

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

## マルチプロセッシング のトラブルシューティング

開始時に プロセス がハングする場合は、[既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})が原因である可能性があります。これを解決するには、`wandb.init` に追加の settings パラメータを追加して、wandb のマルチ プロセッシング プロトコルを変更してみてください。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバル 環境変数 を設定します。

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメータ の最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、高度にスケーラブルな ハイパーパラメータ 検索 プラットフォーム であり、最小限の要件 コード 不動産で W&B experiments に関する興味深い洞察と 可視化 を提供します。 Sweeps は、コーディング要件なしで Hydra projects とシームレスに統合されます。必要なのは、通常どおりスイープするさまざまな パラメータ を記述した 設定 ファイルだけです。

簡単な `sweep.yaml` ファイルの例を次に示します。

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

sweep を呼び出します。

``` bash
wandb sweep sweep.yaml` \
```

W&B は プロジェクト 内に sweep を自動的に作成し、sweep を実行する各マシンで実行する `wandb agent` コマンド を返します。

### Hydra のデフォルトに存在しない パラメータ を渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra は、 コマンド の前に `+` を使用して、デフォルトの 設定 ファイルに存在しない追加の パラメータ を コマンドライン から渡すことをサポートしています。たとえば、次のように呼び出すだけで、いくつかの 値 を持つ追加の パラメータ を渡すことができます。

```bash
$ python program.py +experiment=some_experiment
```

[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) を 設定 する際に行うことと同様に、このような `+` 構成を スイープ することはできません。これを回避するには、デフォルトの空のファイルで experiment パラメータ を初期化し、W&B Sweep を使用して、呼び出しごとにこれらの空の 設定 をオーバーライドします。詳細については、[**この W&B Report**](http://wandb.me/hydra)**をお読みください。**
