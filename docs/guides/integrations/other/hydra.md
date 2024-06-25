---
description: W&B を Hydra と統合する方法
slug: /guides/integrations/hydra
displayed_sidebar: default
---


# Hydra

> [Hydra](https://hydra.cc) は、研究やその他の複雑なアプリケーションの開発を簡素化するオープンソースのPythonフレームワークです。主な特徴は、構成ファイルやコマンドラインを介して、階層的な設定を動的に作成およびオーバーライドできることです。

Hydraを設定管理に使用しながら、W&Bのパワーを活用することができます。

## メトリクスの追跡

`wandb.init` と `wandb.log` を使って通常通りメトリクスを追跡します。ここでは、 `wandb.entity` と `wandb.project` が hydra の設定ファイル内で定義されています。

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## ハイパーパラメーターの追跡

Hydraは設定辞書とインターフェースするためのデフォルトの方法として [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) を使用します。 `OmegaConf` の辞書はプリミティブな辞書のサブクラスではないため、Hydra の `Config` を直接 `wandb.config` に渡すと、ダッシュボード上で予期しない結果になることがあります。 `omegaconf.DictConfig` をプリミティブな `dict` 型に変換してから `wandb.config` に渡す必要があります。

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

### マルチプロセシングのトラブルシューティング

プロセスが開始時にハングする場合は、[この既知の問題](../../track/log/distributed-training.md)が原因である可能性があります。これを解決するには、以下のように wandb のマルチプロセシングプロトコルを追加の設定パラメータで変更してみてください：

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

または、シェルからグローバル環境変数を設定することで：

```
$ export WANDB_START_METHOD=thread
```

## ハイパーパラメーターの最適化

[W&B Sweeps](../../sweeps/intro.md) は非常にスケーラブルなハイパーパラメーター検索プラットフォームで、最小限のコードでW&B Experimentsに関する興味深いインサイトや可視化を提供します。Sweepsはコード不要でHydra Projectsとシームレスに統合できます。必要なのは通常通り、スイープする各種パラメータを記述する設定ファイルだけです。

簡単な例として `sweep.yaml` ファイルは以下のようになります：

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

スイープを呼び出すには：

`wandb sweep sweep.yaml`

これを実行すると、W&Bは自動的にプロジェクト内にスイープを作成し、各マシンでスイープを実行するための `wandb agent` コマンドを返します。

#### Hydra のデフォルトにないパラメーターを渡す <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydraは、デフォルト設定ファイルに存在しない追加パラメーターをコマンドラインから `+` を付けて渡すことができます。例えば、次のようにして追加のパラメーターに値を設定できます：

```
$ python program.py +experiment=some_experiment
```

この `+` 設定をスイープ対象とすることは、Hydra Experimentsの設定と同様にはできません。この回避策として、実験パラメーターをデフォルトの空ファイルで初期化し、各呼び出し時に空の設定をW&B Sweepで上書きする方法があります。詳細は [**このW&B レポート**](http://wandb.me/hydra) をご確認ください。