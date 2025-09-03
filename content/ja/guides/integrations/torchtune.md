---
title: Pytorch チューニングする torchtune
menu:
  default:
    identifier: ja-guides-integrations-torchtune
    parent: integrations
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、大規模言語モデル（LLM）の作成、ファインチューニング、実験プロセスを効率化するために設計された PyTorch ベースのライブラリです。さらに、torchtune は [W&B でのログ記録](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) をサポートしており、トレーニングプロセスの追跡と可視化を強化します。

{{< img src="/images/integrations/torchtune_dashboard.png" alt="" >}}

[torchtune を使用した Mistral 7B のファインチューニング](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) に関する W&B ブログ記事をチェックしてください。

## W&B のログ記録が手の届くところに

{{< tabpane text=true >}}
{{% tab header="Command line" value="cli" %}}

ローンチ時にコマンドライン引数をオーバーライドします:

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="Recipe's config" value="config" %}}

レシピの設定で W&B ログ記録を有効にします:

```yaml
# inside llama3/8B_lora_single_device.yaml
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B メトリックロガーの使用

`metric_logger` セクションを変更して、レシピの設定ファイルで W&B ログ記録を有効にします。`_component_` を `torchtune.utils.metric_logging.WandBLogger` にクラスを変更します。また、`project` 名と `log_every_n_steps` を渡してログ記録の振る舞いをカスタマイズすることもできます。

`wandb.init` メソッドに渡すのと同様に、他の `kwargs` を渡すこともできます。例えば、チームで作業している場合、`WandBLogger` クラスに `entity` 引数を渡してチーム名を指定することができます。

{{< tabpane text=true >}}
{{% tab header="Recipe's Config" value="config" %}}

```yaml
# inside llama3/8B_lora_single_device.yaml
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
  entity: my_project
  job_type: lora_finetune_single_device
  group: my_awesome_experiments
log_every_n_steps: 5
```

{{% /tab %}}

{{% tab header="Command Line" value="cli" %}}

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

## 何がログされますか？

W&B ダッシュボードを探索して、ログされたメトリックを見ることができます。デフォルトでは、W&B は設定ファイルとローンチのオーバーライドからすべてのハイパーパラメーターをログします。

W&B は **Overview** タブで解決された設定をキャプチャします。W&B は YAML 形式で設定を [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) にも保存します。

{{< img src="/images/integrations/torchtune_config.png" alt="" >}}

### ログされたメトリック

各レシピにはそれぞれのトレーニングループがあります。個別のレシピを確認して、そのログされたメトリックを見ることができます。これにはデフォルトで以下が含まれています:

| Metric | 説明 |
| --- | --- |
| `loss` | モデルのロス |
| `lr` | 学習率 |
| `tokens_per_second` | モデルのトークン毎秒 |
| `grad_norm` | モデルの勾配ノルム |
| `global_step` | トレーニングループの現在のステップに対応します。勾配の累積を考慮に入れ、オプティマイザーステップが取られるたびにモデルが更新され、勾配が累積され、`gradient_accumulation_steps` ごとに1回モデルが更新されます。 |

{{% alert %}}
`global_step` はトレーニングステップの数と同じではありません。トレーニングループの現在のステップに対応します。勾配の累積を考慮に入れ、オプティマイザーステップが取られるたびに `global_step` が1増加します。例えば、データローダーに10バッチあり、勾配の累積ステップが2で3エポック走行する場合、オプティマイザーは15回ステップし、この場合 `global_step` は1から15までの範囲になります。
{{% /alert %}}

torchtune の効率的な設計により、カスタムメトリクスを簡単に追加したり、既存のものを変更することができます。対応する [レシピファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を変更し、例えば以下のように `current_epoch` を全エポック数のパーセンテージとして記録するだけで十分です。

```python
# inside `train.py` function in the recipe file
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
これは急速に進化しているライブラリであり、現在のメトリクスは変更される可能性があります。カスタムメトリクスを追加したい場合は、レシピを変更し、対応する `self._metric_logger.*` 関数を呼び出す必要があります。
{{% /alert %}}

## チェックポイントの保存とロード

torchtune ライブラリは様々な [チェックポイントフォーマット](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) をサポートしています。使用しているモデルの出所に応じて、適切な [チェックポインタークラス](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) に切り替えるべきです。

もしモデルのチェックポイントを [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存したい場合は、対応するレシピ内の `save_checkpoint` 関数をオーバーライドするのが最も簡単です。

ここにモデルのチェックポイントを W&B Artifacts に保存するために `save_checkpoint` 関数をオーバーライドする方法の例を示します。

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## Let's save the checkpoint to W&B
    ## depending on the Checkpointer Class the file will be named differently
    ## Here is an example for the full_finetune case
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # description of the model checkpoint
        description="Model checkpoint",
        # you can add whatever metadata you want as a dict
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