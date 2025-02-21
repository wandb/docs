---
title: Pytorch torchtune
menu:
  default:
    identifier: ja-guides-integrations-torchtune
    parent: integrations
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、PyTorch ベースのライブラリで、大規模言語モデル (LLMs) の作成、ファインチューニング、及び実験管理プロセスを簡素化するように設計されています。さらに、torchtune には [W&B を使ったログ](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) が組み込まれており、トレーニングプロセスの追跡と可視化を強化します。

{{< img src="/images/integrations/torchtune_dashboard.png" alt="" >}}

[torchtune を使った Mistral 7B のファインチューニング](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) に関する W&B のブログ投稿をご覧ください。

## W&B ログの活用

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

ローンチ時にコマンドライン引数をオーバーライド：

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="レシピの設定" value="config" %}}

レシピの設定で W&B ログを有効に：

```yaml
# llama3/8B_lora_single_device.yaml 内
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B メトリクスロガーを利用

`metric_logger` セクションを修正して、レシピの設定ファイルで W&B ログを有効にします。`_component_` を `torchtune.utils.metric_logging.WandBLogger` クラスに変更します。また、`project` 名や `log_every_n_steps` を渡してログの振る舞いをカスタマイズすることができます。

他にも [wandb.init]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッドに渡すような `kwargs` を渡すこともできます。例えば、チームで作業している場合、チーム名を指定するために `entity` 引数を `WandBLogger` クラスに渡すことができます。

{{< tabpane text=true >}}
{{% tab header="レシピの設定" value="config" %}}

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

## 何がログされるのか？

ログされたメトリクスを見るために W&B ダッシュボードを探索できます。デフォルトで W&B は設定ファイルから全てのハイパーパラメーターをログし、ローンチのオーバーライドを記録します。

W&B は **Overview** タブで解決された設定をキャプチャします。W&B はまた YAML フォーマットで設定を [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) に保存します。

{{< img src="/images/integrations/torchtune_config.png" alt="" >}}

### ログされたメトリクス

各レシピにはそれぞれのトレーニングループがあります。個々のレシピを確認して、そのログされたメトリクスを確認してください。デフォルトで以下が含まれます：

| メトリクス | 説明 |
| --- | --- |
| `loss` | モデルの損失 |
| `lr` | 学習率 |
| `tokens_per_second` | モデルの毎秒のトークン数 |
| `grad_norm` | モデルの勾配ノルム |
| `global_step` | トレーニングループの現在のステップに対応。勾配累積を考慮し、オプティマイザーステップが行われるたびにモデルが更新され、勾配が累積され、`gradient_accumulation_steps` ごとにモデルが更新される |

{{% alert %}}
`global_step` はトレーニングステップの数と同じではありません。それはトレーニングループの現在のステップに対応します。勾配累積を考慮し、オプティマイザーステップが行われるたびに `global_step` が1増加します。例えば、データローダーに10バッチがあり、勾配累積ステップが2で3エポック実行される場合、オプティマイザーは15回ステップし、この場合 `global_step` は1から15の範囲になります。
{{% /alert %}}

torchtune の洗練されたデザインにより、カスタムメトリクスを簡単に追加したり、既存のメトリクスを修正したりすることができます。対応する [レシピファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を修正するだけで十分です。例えば、以下のように `current_epoch` を全エポック数の割合としてログすることができます：

```python
# レシピファイル内の `train.py` 関数内
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
これは急速に進化するライブラリであり、現在のメトリクスは変更される可能性があります。カスタムメトリクスを追加したい場合は、レシピを修正し、対応する `self._metric_logger.*` 関数を呼び出すべきです。
{{% /alert %}}

## チェックポイントの保存と読み込み

torchtune ライブラリは様々な [チェックポイントフォーマット](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) をサポートしています。使用しているモデルの起源に応じて、適切な [チェックポインタクラ](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)に切り替えるべきです。

[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) にモデルのチェックポイントを保存したい場合、最も簡単な方法は対応するレシピ内で `save_checkpoint` 関数をオーバーライドすることです。

以下は W&B Artifacts にモデルのチェックポイントを保存するために `save_checkpoint` 関数をオーバーライドする方法の例です。

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## W&B にチェックポイントを保存しましょう
    ## Checkpointer クラスに応じてファイル名が異なることがあります
    ## これは full_finetune ケースの例です
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # モデルチェックポイントの説明
        description="Model checkpoint",
        # 任意のメタデータを辞書として追加可能
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