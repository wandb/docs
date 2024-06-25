---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Pytorch torchtune

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb"></CTAButtons>

[torchtune](https://pytorch.org/torchtune/stable/index.html) は、PyTorch ベースのライブラリで、大規模言語モデル（LLM）の作成、微調整、および実験プロセスを簡素化するために設計されています。さらに、torchtune には [W&B を使用したログ作成](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) が組み込まれており、トレーニングプロセスの追跡と可視化を強化します。

![](@site/static/images/integrations/torchtune_dashboard.png)

[Fine-tuning Mistral 7B using torchtune](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0) に関する W&B ブログポストをご覧ください。

## W&B ログ作成を簡単に

<Tabs
  defaultValue="config"
  values={[
    {label: 'Recipe\'s config', value: 'config'},
    {label: 'Command line', value: 'cli'},
  ]}>
  <TabItem value="cli">

コマンドライン引数を上書きして起動するには：

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

  </TabItem>
  <TabItem value="config">

レシピの設定で W&B ログ作成を有効にする
```yaml
# llama3/8B_lora_single_device.yaml の内部
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

  </TabItem>
</Tabs>

## W&B メトリクスロガーを使用する

レシピの設定ファイルで `metric_logger` セクションを修正して W&B ログ作成を有効にします。 `_component_` を `torchtune.utils.metric_logging.WandBLogger` クラスに変更します。また、プロジェクト名の `project` とログ作成間隔の `log_every_n_steps` を渡すこともできます。

他の `kwargs` を [wandb.init](../../ref/python/init.md) メソッドに渡すように `WandBLogger` クラスに渡すこともできます。たとえば、チームで作業している場合は、チーム名を指定するために `entity` 引数を渡すことができます。

<Tabs
  defaultValue="config"
  values={[
    {label: 'Recipe\'s Config', value: 'config'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```shell
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  metric_logger.entity="my_project" \
  metric_logger.job_type="lora_finetune_single_device" \
  metric_logger.group="my_awesome_experiments" \
  log_every_n_steps=5
```
  
  </TabItem>
  <TabItem value="config">

```yaml
# llama3/8B_lora_single_device.yaml の内部
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
  entity: my_project
  job_type: lora_finetune_single_device
  group: my_awesome_experiments
log_every_n_steps: 5
```

  </TabItem>
</Tabs>

## 何がログされるのか？

上記のコマンドを実行すると、W&B のダッシュボードでログされたメトリクスを確認できます。デフォルトで W&B は設定ファイルと起動時のオーバーライドのすべてのハイパーパラメーターを取得します。

W&B は **Overview** タブで解決された設定をキャプチャします。また、W&B は設定を YAML として [Files タブ](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files) に保存します。

![](@site/static/images/integrations/torchtune_config.png)

### ログされたメトリクス

各レシピには独自のトレーニングループがあるため、各レシピごとにログされるメトリクスを確認してください。デフォルトでログされるメトリクスは以下の通りです：

| メトリクス | 説明 |
| --- | --- |
| `loss` | モデルの損失 |
| `lr` | 学習率 |
| `tokens_per_second` | モデルの毎秒トークン数 |
| `grad_norm` | モデルの勾配ノルム |
| `global_step` | トレーニングループの現在のステップに対応します。勾配の累積を考慮に入れ、オプティマイザーステップが取られるたびにモデルが更新され、勾配が累積され、`gradient_accumulation_steps`ごとにモデルが一度更新されます |

:::info
`global_step` はトレーニングステップ数と同じではありません。トレーニングループの現在のステップに対応します。オプティマイザーステップが取られるたびに `global_step` は 1 つ増加します。例えば、データローダーに 10 バッチがあり、勾配累積ステップが 2 で、3 エポックを実行する場合、オプティマイザーは 15 回ステップし、この場合 `global_step` は 1 から 15 の範囲になります。
:::

torchtune のシンプルな設計により、カスタムメトリクスの追加や既存のメトリクスの修正が容易です。対応する [レシピファイル](https://github.com/pytorch/torchtune/tree/main/recipes) を修正するだけで済みます。たとえば、次のように、総エポック数の割合として `current_epoch` をログすることができます：

```python
# レシピファイル内の `train.py` 関数
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

:::info
これは急速に進化しているライブラリで、現在のメトリクスは変更される可能性があります。カスタムメトリクスを追加したい場合は、レシピを修正し、対応する `self._metric_logger.*` 関数を呼び出す必要があります。
:::

## チェックポイントの保存と読み込み

torchtune ライブラリは、さまざまな [チェックポイント形式](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) をサポートします。使用しているモデルの出所に応じて、適切な [checkpointer クラス](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html) に切り替える必要があります。

モデルチェックポイントを [W&B Artifacts](../artifacts/intro.md) に保存したい場合、最も簡単な解決策は、対応するレシピ内の `save_checkpoint` 関数を上書きすることです。

モデルチェックポイントを W&B Artifacts に保存する方法の例は以下の通りです：

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## W&B にチェックポイントを保存しましょう
    ## Checkpointer クラスに依存してファイル名が異なります
    ## Full_finetune ケースの例はこちら
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # モデルチェックポイントの説明
        description="Model checkpoint",
        # 任意のメタデータを辞書形式で追加可能
        metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
        }
    )
    wandb_artifact.add_file(checkpoint_file)
    wandb.log_artifact(wandb_artifact)
```