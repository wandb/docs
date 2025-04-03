---
title: Hydra
description: W&B と Hydra を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、 研究 および他の複雑な アプリケーション の開発を簡素化するオープンソースの Python フレームワーク です。主な機能は、構成ファイルと コマンドライン を介して、構成を構成してオーバーライドすることにより、階層的な構成を動的に作成できることです。

W&B のパワーを利用しながら、構成管理に Hydra を引き続き使用できます。

## メトリクス の 追跡

`wandb.init` と `wandb.log` を使用して、通常どおりに メトリクス を 追跡 します。ここでは、 `wandb.entity` と `wandb.project` は hydra 構成ファイル内で定義されています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメーター の 追跡

Hydra は、構成 辞書 とのインターフェースをとるためのデフォルトの方法として [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を使用します。 `OmegaConf` の 辞書 は、プリミティブ 辞書 のサブクラスではないため、Hydra の `Config` を `wandb.config` に直接渡すと、 ダッシュボード で予期しない結果が生じます。 `omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.config` に渡す必要があります。

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

## マルチプロセッシング の トラブルシューティング

開始時に プロセス がハングする場合は、[この既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) が原因である可能性があります。これを解決するには、次のいずれかとして、`wandb.init` に追加の settings パラメータ を追加して、wandb の マルチプロセッシング プロトコル を変更してみてください。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルから グローバル 環境変数 を設定します。

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーター の 最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、高度にスケーラブルな ハイパーパラメーター 検索 プラットフォーム であり、最小限の要件コードで W&B の 実験 に関する興味深い洞察と 可視化 を提供します。 Sweeps は、コーディング要件なしで Hydra プロジェクト とシームレスに統合されます。必要なのは、通常どおりにスイープするさまざまな パラメータ を記述した構成ファイルだけです。

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

### Hydra の デフォルト に存在しない パラメータ を渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra は、 コマンド の前に `+` を使用して、デフォルト の構成ファイルに存在しない追加の パラメータ を コマンドライン から渡すことをサポートしています。たとえば、次のように呼び出すだけで、いくつかの 値 を持つ追加の パラメータ を渡すことができます。

```bash
$ python program.py +experiment=some_experiment
```

[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) を構成するときと同様に、このような `+` 構成を スイープ することはできません。これを回避するには、デフォルト の空のファイルで experiment パラメータ を初期化し、W&B Sweep を使用して、呼び出しごとにこれらの空の構成をオーバーライドします。詳細については、[**この W&B Report**](http://wandb.me/hydra) をお読みください。**
