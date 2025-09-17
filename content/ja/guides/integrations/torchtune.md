---
title: Pytorch torchtune
menu:
  default:
    identifier: ja-guides-integrations-torchtune
    parent: integrations
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、PyTorch ベースで LLM（大規模言語モデル）の作成、ファインチューニング、実験を効率化するために設計されたライブラリです。さらに torchtune には [W&B へのロギング](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) が標準で組み込まれており、トレーニング プロセスの追跡と可視化が強化されています。

{{< img src="/images/integrations/torchtune_dashboard.png" alt="torchtune のトレーニング ダッシュボード" >}}

[torchtune で Mistral 7B をファインチューニング](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) に関する W&B ブログもご覧ください。

## W&B ロギングをすぐに使う

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

起動時にコマンドライン引数を上書きします:

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="レシピの設定" value="config" %}}

レシピの設定で W&B ロギングを有効化します:

```yaml
# llama3/8B_lora_single_device.yaml の中
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B のメトリクス ロガーを使う

レシピの設定ファイルで `metric_logger` セクションを変更して W&B ロギングを有効化します。`_component_` を `torchtune.utils.metric_logging.WandBLogger` クラスに変更してください。ロギングの振る舞いをカスタマイズするために、`project` 名や `log_every_n_steps` も渡せます。

[wandb.init()]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡せるのと同じ任意の `kwargs` も利用できます。たとえばチームで作業している場合は、チーム名を指定するために `WandBLogger` クラスに `entity` 引数を渡してください。

{{< tabpane text=true >}}
{{% tab header="レシピの設定" value="config" %}}

```yaml
# llama3/8B_lora_single_device.yaml の中
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
  entity: my_project
  job_type: lora_finetune_single_device
  group: my_awesome_experiments
log_every_n_steps: 5
```

{{% /tab %}}

{{% tab header="コマンドライン" value="cli" %}}

```shell
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  metric_logger.entity="my_project" \
  metric_logger.job_type="lora_finetune_single_device" \
  metric_logger.group="my_awesome_experiments" \
  log_every_n_steps=5
```

{{% /tab %}}
{{< /tabpane >}}

## 何がログされるか

W&B ダッシュボードでログされたメトリクスを確認できます。デフォルトで、W&B は設定ファイルのすべてのハイパーパラメーターと、起動時に上書きした値をログします。

W&B は解決済みの設定を **Overview** タブに記録します。設定は YAML 形式でも [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) に保存されます。

{{< img src="/images/integrations/torchtune_config.png" alt="torchtune の設定" >}}

### ログされるメトリクス

各レシピは独自のトレーニング ループを持っています。ログされるメトリクスはレシピごとに確認してください。以下は既定で含まれるものです:

| メトリクス | 説明 |
| --- | --- |
| `loss` | モデルの損失 |
| `lr` | 学習率 |
| `tokens_per_second` | モデルの毎秒トークン数 |
| `grad_norm` | モデルの勾配ノルム |
| `global_step` | 現在のトレーニング ループにおけるステップに対応します。勾配蓄積を考慮し、基本的にオプティマイザーのステップが実行されるたびにモデルが更新され、勾配が蓄積され、`gradient_accumulation_steps` ごとに 1 回モデルが更新されます。 |

{{% alert %}}
`global_step` はトレーニング ステップ数そのものとは同じではありません。これは現在のトレーニング ループにおけるステップに対応します。勾配蓄積を考慮し、基本的にオプティマイザーのステップが実行されるたびに `global_step` は 1 ずつ増加します。たとえば、データローダーのバッチ数が 10、勾配蓄積ステップが 2、エポック数が 3 の場合、オプティマイザーは 15 回ステップを実行します。このケースでは `global_step` は 1 から 15 の範囲になります。
{{% /alert %}}

torchtune のシンプルな設計により、カスタム メトリクスの追加や既存メトリクスの変更が容易です。対応する [レシピ ファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を変更すれば十分です。たとえば、トータル エポック数に対する進捗率として `current_epoch` をログするには次のようにします:

```python
# レシピ ファイルの `train.py` 関数内
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
このライブラリは急速に進化しています。現在のメトリクスは変更される可能性があります。カスタム メトリクスを追加したい場合は、レシピを変更し、対応する `self._metric_logger.*` 関数を呼び出してください。
{{% /alert %}}

## チェックポイントの保存と読み込み

torchtune ライブラリは、さまざまな[チェックポイント形式](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)をサポートしています。使用しているモデルの出自に応じて、適切な[Checkpointer クラス](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)に切り替えてください。

モデルのチェックポイントを [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存したい場合は、対応するレシピ内の `save_checkpoint` 関数をオーバーライドするのが最もシンプルです。

以下は、`save_checkpoint` 関数をオーバーライドしてモデルのチェックポイントを W&B Artifacts に保存する例です。

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## チェックポイントを W&B に保存しましょう
    ## 使用する Checkpointer クラスに応じてファイル名は異なります
    ## これは full_finetune の例です
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # モデルのチェックポイントの説明
        description="Model checkpoint",
        # 辞書として任意のメタデータを追加できます
        metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
        },
    )
    wandb_artifact.add_file(checkpoint_file)
    wandb.log_artifact(wandb_artifact)
```