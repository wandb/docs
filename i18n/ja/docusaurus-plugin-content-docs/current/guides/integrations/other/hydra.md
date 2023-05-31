---
slug: /guides/integrations/hydra
description: How to integrate W&B with Hydra.
displayed_sidebar: ja
---

# Hydra

> [Hydra](https://hydra.cc) は、研究や複雑なアプリケーションの開発を簡素化するオープンソースのPythonフレームワークです。主な特徴は、コンフィグファイルやコマンドラインを介して階層構造の設定を動的に作成、上書きできる機能です。

Hydraを設定管理のために使用しながら、W&Bの力を利用することができます。

## メトリクスをトラッキング

`wandb.init` と `wandb.log` を使ってメトリクスをトラッキングします。ここでは、`wandb.entity` と `wandb.project` は、Hydraの設定ファイル内で定義されています。

```python
import wandb

@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメーターをトラッキング

Hydraでは、[omegaconf](https://omegaconf.readthedocs.io/en/2.1\_branch/)がデフォルトの設定ディクショナリとして使用されます。`OmegaConf` のディクショナリはプリミティブなディクショナリのサブクラスではないため、Hydraの `Config` を直接 `wandb.config` に渡すと、ダッシュボード上で予期しない結果が発生します。`omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.config` に渡す必要があります。

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
### マルチプロセッシングのトラブルシューティング

プロセスが開始されたときに停止してしまう場合は、[この既知の問題](../../track/log/distributed-training.md)が原因である可能性があります。これを解決するには、wandbのマルチプロセッシングプロトコルを変更してみてください。`wandb.init`に追加の設定パラメータを追加することで変更できます。

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバル環境変数を設定することもできます。

```
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメータを最適化する

[W&Bスイープ](../../sweeps/intro.md)は、高いスケーラビリティを持ったハイパーパラメーターサーチプラットフォームであり、W&B実験に関する興味深い洞察と可視化を、コードの最小要件を満たした範囲で提供します。 スイープは、Hydraプロジェクトとシームレスに統合され、コーディングの要件はありません。 必要なのは、スイープするさまざまなパラメータを通常どおり記述した設定ファイルだけです。

簡単な例として、`sweep.yaml`ファイルは以下のようになります。

```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test/accuracy
parameters:
  dataset:
    values: [mnist, cifar10]
コマンド:

  - ${env}

  - python

  - ${program}

  - ${args_no_hyphens}

```

スイープを起動するには：

`wandb sweep sweep.yaml`\
\
``W&Bは、このコマンドを呼び出すと、プロジェクト内に自動的にスイープを作成し、スイープを実行したい各マシン上で実行するための `wandb agent` コマンドを返します。

#### Hydraのデフォルトにないパラメーターを渡す <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydraは、コマンドラインを通じてデフォルトの設定ファイルに存在しない追加のパラメーターを渡すことができ、コマンドの前に`+`を使用します。例えば、次のように呼び出すだけで、いくつかの値を持つ追加のパラメーターを渡すことができます。

```
$ python program.py +experiment=some_experiment
```

このような`+`設定は、[Hydra Experiments](https://hydra.cc/docs/patterns/configuring\_experiments/)の設定時と同様に、スイープできません。これを回避するために、実験パラメーターをデフォルトの空のファイルで初期化し、W&Bスイープを使用して各呼び出しで空の設定を上書きできます。詳細については、[**このW&Bレポート**](http://wandb.me/hydra) をご覧ください。