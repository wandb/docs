---
description: W&B を PyTorch torchtune と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-torchtune
    parent: integrations
title: Pytorch torchtune
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、PyTorch をベースとしたライブラリで、大規模言語モデル（LLM）の作成、ファインチューニング、実験管理を効率化することを目的としています。さらに、torchtune は [W&B でのロギング](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html)を標準サポートしており、トレーニングプロセスのトラッキングや可視化を強化できます。

{{< img src="/images/integrations/torchtune_dashboard.png" alt="TorchTune training dashboard" >}}

W&B ブログの [Fine-tuning Mistral 7B using torchtune](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) もぜひご覧ください。

## W&B ロギングをすぐに使う

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

起動時にコマンドライン引数を上書きできます：

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="レシピの設定" value="config" %}}

レシピの設定ファイルで W&B ロギングを有効にします：

```yaml
# llama3/8B_lora_single_device.yaml 内
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B メトリクスロガーの利用

`metric_logger` セクションを書き換えることで、レシピの設定ファイル内で W&B ロギングを有効にできます。`_component_` を `torchtune.utils.metric_logging.WandBLogger` クラスに変更しましょう。また、`project` 名や `log_every_n_steps` でロギングの挙動もカスタマイズ可能です。

他にも、[wandb.init()]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) メソッドに渡せるあらゆる `kwargs` を `WandBLogger` クラスに指定できます。たとえば、チームで作業している場合は `entity` 引数でチーム名を指定可能です。

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

## どんな情報がログされる？

W&B のダッシュボードで、ログされたメトリクスを確認できます。デフォルトでは、設定ファイルや起動時に上書きした全てのハイパーパラメーターが W&B に記録されます。

W&B は解決済み設定を **Overview** タブで表示し、設定は YAML 形式で [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) に保存します。

{{< img src="/images/integrations/torchtune_config.png" alt="TorchTune configuration" >}}

### ログされるメトリクス

各レシピごとに独自のトレーニングループがあり、それぞれのレシピでどんなメトリクスが記録されるか確認できますが、デフォルトでは以下が含まれています：

| Metric | 説明 |
| --- | --- |
| `loss` | モデルの損失値 |
| `lr` | 学習率 |
| `tokens_per_second` | 1 秒あたりに処理したトークン数 |
| `grad_norm` | 勾配ノルム |
| `global_step` | トレーニングループ内での現在のステップ。勾配の累積を考慮し、オプティマイザーが step を実行するたび、モデルが更新され、勾配が累積され、`gradient_accumulation_steps` ごとにモデルが 1 度更新される |

{{% alert %}}
`global_step` はトレーニングステップ数と同じではありません。これはトレーニングループ内の現在の step を示し、勾配の累積を考慮します。基本的にオプティマイザーが step を実行するたび `global_step` が 1 ずつ増えます。たとえば、dataloader が 10 バッチ、gradient accumulation steps が 2、エポック数が 3 の場合、オプティマイザーは計 15 回ステップし、この場合 `global_step` は 1 から 15 まで進みます。
{{% /alert %}}

torchtune のシンプルな設計により、カスタムメトリクスの追加や既存メトリクスの変更も容易です。対応する [レシピファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を編集するだけで、例えば `current_epoch` をエポック全体のパーセンテージとしてログすることもできます：

```python
# レシピファイルの `train.py` 関数内
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
このライブラリは急速に進化中のため、現状のメトリクスは変更となる可能性があります。カスタムメトリクスを追加したい場合は、レシピを編集して該当する `self._metric_logger.*` 関数を呼んでください。
{{% /alert %}}

## チェックポイントの保存と読み込み

torchtune ライブラリは [様々なチェックポイント形式](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)をサポートしています。使っているモデルの種類に応じて、適切な [Checkpointer クラス](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)の利用が推奨されます。

モデルのチェックポイントを [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存する場合は、対応するレシピ内で `save_checkpoint` 関数を上書きするのがシンプルな解決策です。

下記は、W&B Artifacts にモデルチェックポイントを保存する `save_checkpoint` 関数のオーバーライド例です。

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## チェックポイントを W&B に保存しましょう
    ## Checkpointer クラスによってファイル名は異なります
    ## ここでは full_finetune 用のサンプルです
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # モデルチェックポイントの説明
        description="Model checkpoint",
        # 任意のメタデータ（辞書型）を追加可能
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
