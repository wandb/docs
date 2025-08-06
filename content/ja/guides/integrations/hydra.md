---
title: Hydra
description: W&B を Hydra と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、研究やその他の複雑なアプリケーション開発をシンプルにしてくれる、オープンソースの Python フレームワークです。主な特徴は、階層的な設定を動的に作成し、設定ファイルやコマンドラインから上書きできることです。

W&B の強力な機能を活かしながら、設定管理には引き続き Hydra を利用できます。

## メトリクスのトラッキング

`wandb.init()` と `wandb.Run.log()` を使って、通常通りメトリクスをトラッキングできます。ここでは `wandb.entity` と `wandb.project` を hydra の設定ファイルで定義しています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
      run.log({"loss": loss})  # ロスをログする
```

## ハイパーパラメーターのトラッキング

Hydra はデフォルトで [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を利用し、設定の辞書操作を行います。`OmegaConf` の辞書は、通常の辞書（dict）のサブクラスではないため、Hydra の `Config` をそのまま `wandb.Run.config` に渡すとダッシュボードで想定外の挙動になることがあります。`omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.Run.config` へ渡す必要があります。

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
  with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
    # 設定を dict 型に変換して config に渡す
    run.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    run.log({"loss": loss})  # ロスをログ
    model = Model(**run.config.model.configs)
```

## マルチプロセスのトラブルシュート

プロセス実行時にハングする場合は、[こちらの既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) が原因かもしれません。その場合、`wandb.init()` に追加の settings パラメータを渡して W&B のマルチプロセスプロトコルを変更できます。たとえば：

```python
wandb.init(settings=wandb.Settings(start_method="thread"))  # マルチプロセス設定を変更
```

または、シェルからグローバル環境変数を設定します：

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーターの最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、大規模なハイパーパラメーター探索を柔軟に行えるプラットフォームです。最小限のコードで W&B 実験の可視化やインサイトを得られます。Sweeps は Hydra プロジェクトともコード不要でシームレスに統合できます。必要なのは、通常通りスイープしたいパラメーターを記述した設定ファイルだけです。

シンプルな例としての `sweep.yaml` ファイル：

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

スイープを実行するには：

``` bash
wandb sweep sweep.yaml \
```

W&B はプロジェクト内に自動でスイープを作成し、実行したいマシンごとに `wandb agent` コマンドを返してくれます。

### Hydra のデフォルトにないパラメータを渡したい場合

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra では、デフォルト設定ファイルに存在しない追加パラメーターも、コマンドラインで `+` をつけて渡せます。例えば、以下のように追加のパラメータを簡単に指定できます：

```bash
$ python program.py +experiment=some_experiment
```

ただし、このような `+` 設定は [Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) のようにスイープ対象として利用できません。回避策としては、あらかじめ空のファイルなどで experiment パラメーターを初期化し、W&B Sweep で実行ごとにその空の設定を上書きする方法があります。詳細は [この W&B Report](https://wandb.me/hydra) をご覧ください。