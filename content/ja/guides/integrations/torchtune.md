---
title: Pytorch torchtune
menu:
  default:
    identifier: torchtune
    parent: integrations
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、大規模言語モデル（LLM）の作成、ファインチューニング、実験プロセスを効率化するための PyTorch ベースのライブラリです。さらに、torchtune には [W&B でのログ機能](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) が内蔵されており、トレーニングプロセスの追跡と可視化を強化します。

{{< img src="/images/integrations/torchtune_dashboard.png" alt="TorchTune training dashboard" >}}

[Fine-tuning Mistral 7B using torchtune](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) に関する W&B ブログポストもご覧ください。

## W&B ロギングを手軽に利用

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

起動時にコマンドライン引数を上書きします：

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="レシピの config" value="config" %}}

レシピの config で W&B ログを有効にします：

```yaml
# llama3/8B_lora_single_device.yaml 内
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B メトリクスロガーを使う

`metric_logger` セクションを変更することで、レシピの config ファイルで W&B ロギングを有効化できます。`_component_` を `torchtune.utils.metric_logging.WandBLogger` クラスに変更してください。`project` 名や `log_every_n_steps` を渡すことで、ロギングの挙動をカスタマイズ可能です。

また、[wandb.init()]({{< relref "/ref/python/sdk/functions/init.md" >}}) メソッドと同様に、他の `kwargs` も指定できます。たとえば、チームで作業している場合は、`entity` 引数でチーム名を `WandBLogger` クラスに渡すことができます。

{{< tabpane text=true >}}
{{% tab header="レシピの Config" value="config" %}}

```yaml
# llama3/8B_lora_single_device.yaml 内
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

## どんな内容がログされる？

W&B ダッシュボードで記録されたメトリクスを確認できます。デフォルトでは、config ファイルのすべてのハイパーパラメーターや launch 時のオーバーライドがログされます。

W&B は解決済みの config を **Overview** タブに記録します。config は [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) に YAML 形式でも保存されます。

{{< img src="/images/integrations/torchtune_config.png" alt="TorchTune configuration" >}}

### ログされるメトリクス

各レシピには独自のトレーニングループがあります。それぞれのレシピで、どのメトリクスが記録されるかをご確認ください。デフォルトで含まれるのは以下の通りです：

| メトリクス | 説明 |
| --- | --- |
| `loss` | モデルの損失値 |
| `lr` | 学習率 |
| `tokens_per_second` | 1秒あたりのトークン数 |
| `grad_norm` | モデルの勾配ノルム |
| `global_step` | トレーニングループの現在のステップ。勾配の累積（gradient accumulation）を考慮し、オプティマイザーのステップが行われるたびにモデルが更新されます。`gradient_accumulation_steps` ごとに1回モデルの更新が行われます |

{{% alert %}}
`global_step` はトレーニングステップ数と同じではありません。トレーニングループ中の現在のステップを表します。勾配の累積（gradient accumulation）も考慮されています。つまり、オプティマイザーのステップが実行されるたびに `global_step` が 1 進みます。たとえば、dataloader のバッチが 10、gradient accumulation step が 2、エポック数が 3 の場合、オプティマイザーのステップは 15 回実行され、この場合 `global_step` は 1 から 15 までとなります。
{{% /alert %}}

torchtune のシンプルな設計により、カスタムメトリクスの追加や既存メトリクスの変更も簡単です。該当する [レシピファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を修正すれば十分です。例えば、エポック進捗をパーセンテージで表す `current_epoch` をログする場合は、以下のように記述できます：

```python
# レシピファイルの `train.py` 関数内
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
このライブラリは非常に活発に開発が進行しているため、現在のメトリクスは今後変更される可能性があります。カスタムメトリクスを追加したい場合は、レシピを修正し、該当する `self._metric_logger.*` 関数を呼び出してください。
{{% /alert %}}

## チェックポイントの保存とロード

torchtune ライブラリは様々な [チェックポイント形式](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) に対応しています。利用しているモデルの種類によって適した [checkpointer クラス](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) に切り替える必要があります。

モデルのチェックポイントを [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) に保存したい場合は、該当するレシピ内の `save_checkpoint` 関数を上書きするのが最も簡単です。

以下は、`save_checkpoint` 関数を上書きして W&B Artifacts にモデルのチェックポイントを保存する例です。

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## チェックポイントを W&B に保存しましょう
    ## Checkpointer クラスによってファイル名は異なります
    ## こちらは full_finetune 用の例です
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # モデルチェックポイントの説明
        description="Model checkpoint",
        # 必要なメタデータ（辞書型）を任意に追加可能
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