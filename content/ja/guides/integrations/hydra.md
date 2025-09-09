---
title: Hydra
description: W&B を Hydra と 統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-hydra
    parent: integrations
weight: 150
---

> [Hydra](https://hydra.cc) は、研究やその他の複雑な アプリケーション の開発を簡素化するオープンソースの Python フレームワークです。主な特徴は、構成要素を組み合わせて動的に階層的な 設定 を作り、設定ファイルや コマンドライン から上書きできる点です。

W&B の強力さを活かしつつ、設定管理には引き続き Hydra を使えます。

## メトリクスを記録

`wandb.init()` と `wandb.Run.log()` を使って、通常どおり メトリクス を記録します。ここでは、`wandb.entity` と `wandb.project` を Hydra の 設定ファイル 内で定義しています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
      run.log({"loss": loss})
```

## ハイパーパラメーターを記録

Hydra は 設定 の 辞書 とやり取りするデフォルトの手段として [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を使用します。`OmegaConf` の 辞書 はプリミティブな辞書のサブクラスではないため、Hydra の `Config` をそのまま `wandb.Run.config` に渡すと ダッシュボード 上で想定外の結果になります。`omegaconf.DictConfig` を、`wandb.Run.config` に渡す前にプリミティブな `dict` 型へ変換する必要があります。

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
  with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
    run.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    run.log({"loss": loss})
    model = Model(**run.config.model.configs)
```

## マルチプロセスのトラブルシュート

起動時に プロセス がハングする場合は、[既知の問題]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) が原因かもしれません。これを解決するには、次のいずれかの方法で wandb のマルチプロセッシング プロトコルを変更してみてください。`wandb.init()` に 追加の 設定 パラメータを渡す方法:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバルな 環境 変数を設定する方法:

```bash
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーターを最適化

[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は高いスケーラビリティを備えたハイパーパラメーター探索 プラットフォーム で、最小限の コード 変更で W&B の実験に関する有益なインサイトと 可視化 を提供します。Sweeps はコード変更不要で Hydra のプロジェクト とシームレスに統合できます。必要なのは、通常どおり sweep 対象の各 パラメータ を記述した 設定 ファイルだけです。

簡単な `sweep.yaml` の例:

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

sweep を実行します:

``` bash
wandb sweep sweep.yaml` \
```

W&B はあなたの Project 内に自動で sweep を作成し、各マシンで sweep を実行するための `wandb agent` コマンドを返します。

### Hydra の defaults に存在しないパラメータを渡す

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra は、デフォルトの 設定 ファイルに存在しない 追加の パラメータを、コマンドの前に `+` を付けることで コマンドライン から渡すことをサポートしています。例えば、次のように呼び出すだけで任意の パラメータ を追加できます:

```bash
$ python program.py +experiment=some_experiment
```

[Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/) の設定時に行うのと同様に、このような `+` 構成を sweep の対象にすることはできません。回避策として、experiment パラメータをデフォルトの空ファイルで初期化し、各呼び出しで W&B Sweep を使ってその空の 設定 を上書きしてください。詳しくは、[この W&B Report](https://wandb.me/hydra) をご覧ください。