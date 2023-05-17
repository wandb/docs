import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MMDetection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/wandb/examples/blob/master/colabs/mmdetection/Train\_an\_Object\_Detection%2BSemantic\_Segmentation\_Model\_with\_MMDetection\_and\_W%26B.ipynb)

[MMDetection](https://github.com/open-mmlab/mmdetection/)は、PyTorchに基づくオープンソースのオブジェクト検出ツールボックスであり、[OpenMMLab](https://openmmlab.com/)の一部です。組み立てやすいモジュール式のAPIデザインを提供しており、カスタムオブジェクト検出やセグメンテーション開発フローを簡単に構築することができます。

[Weights and Biases](https://wandb.ai/site)は、MMDetectionに専用の`MMDetWandbHook`を介して直接統合されており、次のことができます。

✅ トレーニングと評価のメトリクスをロギングする。

✅ バージョン管理されたモデルのチェックポイントをロギングする。

✅ バージョン管理された検証データセットと正解のバウンディングボックスをロギングする。

✅ モデル予測のロギングと可視化。

## :fire: はじめに

### wandbにサインアップしてログインする

a) [**無料アカウントのサインアップ**](https://wandb.ai/site)

b) `wandb` ライブラリをpipでインストール

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインして、[**Authorizeページ**](https://wandb.ai/authorize)で**APIキーを見つけます。**

もし、Weights and Biasesを初めて使う場合は、[クイックスタート](../../quickstart.md)をチェックしてみてください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### `MMDetWandbHook`を使う方法

Weights and Biasesを使い始めるには、MMDetectionの`log_config`メソッドに`MMDetWandbHook`を追加して、設定システムを利用します。
:::info
`MMDetWandbHook`は、[MMDetection v2.25.0](https://twitter.com/OpenMMLab/status/1532193548283432960?s=20&t=dzBiKn9dlNdrvK8e_q0zfQ)以降でサポートされています。
:::

```python
import wandb
...

config_file = 'mmdetection/configs/path/to/config.py'
cfg = Config.fromfile(config_file)

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'mmdetection'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100,
         bbox_score_thr=0.3)]
```

| Name                      | Description                                                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init_kwargs`             | (`dict`) W&B runを初期化するためにwandb.initに渡される辞書。                                                                                                          |
| `interval`                | (`int`) ロギング間隔（k回のイテレーションごと）。デフォルトは`50`。                                                                                                        |
| `log_checkpoint`          | (`bool`) チェックポイント間隔ごとにチェックポイントをW&Bアーティファクトとして保存します。各バージョンがチェックポイントであるモデルバージョニングに使用してください。デフォルトは`False`。     |
| `log_checkpoint_metadata` | (`bool`) 検証データで計算された評価メトリクスを、現在のエポックと共に、そのチェックポイントのメタデータとしてログします。デフォルトは `True`。 |
| `num_eval_images`         | (`int`) ログされる検証画像の数。ゼロの場合、評価はログされません。デフォルトは `100`。                                                       |
| `bbox_score_thr`          | (`float`) バウンディングボックススコアの閾値。デフォルトは `0.3`。                                                                                                         |
### :chart\_with\_upwards\_trend: メトリクスのログ

`MMDetWandbHook`の`init_kwargs`引数を使って、トレーニングと評価のメトリクスのトラッキングを開始します。この引数は、キーと値のペアを持つディクショナリを受け取り、これが`wandb.init`に渡されて、runがどのプロジェクトにログされるかやその他のrunの機能を制御します。

```
init_kwargs={
    'project': 'mmdetection',
    'entity': 'my_team_name',
    'config': {'lr': 1e-4, 'batch_size':32},
    'tags': ['resnet50', 'sgd'] 
}
```

wandb.initの全ての引数については、[こちら](https://docs.wandb.ai/ref/python/init)をご覧ください。

![](@site/static/images/integrations/log_metrics.gif)

### :checkered\_flag: チェックポイント

`MMDetWandbHook`の`log_checkpoint=True`引数を使って、これらのチェックポイントを[W&B Artifacts](../artifacts/intro.md)として確実に保存できます。この機能は、MMCVの[`CheckpointHook`](https://mmcv.readthedocs.io/en/latest/api.html?highlight=CheckpointHook#mmcv.runner.CheckpointHook)に依存しており、チェックポイントのモデルを定期的に保存します。期間は`checkpoint_config.interval`で決定されます。

:::info
すべてのW&Bアカウントには、データセットとモデル用に100 GBの無料ストレージが付属しています。
:::

![チェックポイントが左側のペインに異なるバージョンとして表示されます。ファイルタブからモデルをダウンロードするか、APIを使ってプログラムでダウンロードできます。](/images/integrations/mmdetection_checkpointing.png)

### :mega: メタデータ付きチェックポイント

`log_checkpoint_metadata`が`True`の場合、すべてのチェックポイントバージョンには、関連するメタデータが付いています。この機能は、`CheckpointHook`、`EvalHook`、または`DistEvalHook`に依存します。メタデータは、チェックポイント間隔が評価間隔で割り切れる場合にのみログされます。
![メタデータタブの下に表示されるログされたメタデータ。](@site/static/images/integrations/mmdetection_checkpoint_metadata.png)

### データセットとモデル予測の可視化

データセットや特にモデル予測をインタラクティブに可視化する能力は、より良いモデルの構築やデバッグに役立ちます。`MMDetWandbHook`を使用することで、W&B Tablesで検証データをログし、モデル予測のバージョン管理されたW&B Tablesを作成できるようになります。

`num_eval_images`引数は、W&B Tablesとしてログされる検証サンプルの数を制御します。以下の点に注意してください。

* `num_eval_images=0`の場合、検証データおよびモデル予測はログされません。
* [`mmdet.core.train_detector`](https://mmdetection.readthedocs.io/en/latest/\_modules/mmdet/apis/train.html?highlight=train\_detector) APIの`validate=False`の場合、検証データおよびモデル予測はログされません。
* `num_eval_images`が検証サンプルの総数よりも大きい場合、完全な検証データセットがログされます。

<!-- ![](/images/integrations/mmdetection_visualize.gif) -->

:::info
`val_data`は一度だけアップロードされます。`run_<id>_pred`テーブルおよびその後の実行は、アップロードされたデータへの参照を使用してメモリを節約します。`val_data`の新しいバージョンは、変更された場合にのみ作成されます。
:::

## 次のステップ

カスタムデータセットでインスタンスセグメンテーションモデル（Mask R-CNN）をトレーニングしたい場合は、[MMDetectionとWeights & Biasesを使った方法](https://wandb.ai/ayush-thakur/mmdetection/reports/How-to-Use-Weights-Biases-with-MMDetection--VmlldzoyMTM0MDE2) のW&Bレポートを[Fully Connected](https://wandb.ai/fully-connected)でチェックしてください。

Weights & Biasesとの統合に関する質問や問題はありますか？[MMDetection githubリポジトリ](https://github.com/open-mmlab/mmdetection)にイシューをオープンしていただければ、対応して回答を提供いたします。:)