---
description: Create custom aliases for W&B Artifacts.
displayed_sidebar: ja
---

# カスタムエイリアスの作成

<head>
    <title>Artifact用のカスタムエイリアスを作成する。</title>
</head>
エイリアスを特定のバージョンへのポインタとして使用します。デフォルトでは、`Run.log_artifact`はログされたバージョンに`latest`エイリアスを追加します。

アーティファクトを初めてログするときに、アーティファクトに`v0`というバージョンが作成され、それがアーティファクトに関連付けられます。Weights & Biasesは、同じアーティファクトに再度ログを記録すると、内容をチェックサムします。アーティファクトが変更された場合、Weights & Biasesは新しいバージョン`v1`を保存します。

例えば、トレーニングスクリプトで最新のデータセットバージョンを引き出したい場合は、アーティファクトを使用するときに`latest`を指定します。以下のコード例では、`latest`というエイリアスを持つ、`bike-dataset`というデータセットアーティファクトの最近のバージョンをダウンロードする方法を示しています：

```python
import wandb

run = wandb.init(project='<example-project>')

アーティファクト = run.use_artifact('bike-dataset:latest')

アーティファクト.download()
```
アーティファクトのバージョンにカスタムエイリアスを適用することもできます。例えば、モデルのチェックポイントがメトリックAP-50で最も優れていることを示すために、モデルのアーティファクトをログするときに文字列`'best-ap50'`をエイリアスとして追加できます。

```python
artifact = wandb.Artifact(
    'run-3nq3ctyy-bike-model', 
    type='model'
    )  
artifact.add_file('model.h5')
run.log_artifact(artifact, aliases=['latest','best-ap50'])
```