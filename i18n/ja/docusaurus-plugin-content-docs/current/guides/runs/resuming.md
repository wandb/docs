---
description: 一時停止または終了したW&B Runを再開する
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Runの再開

<head>
  <title>W&B Run の再開</title>
</head>

`wandb.init()`に`resume=True`を渡すことで、自動的にwandbのRunを再開させることができます。プロセスが正常に終了しない場合、次回実行時にwandbは最後のステップからログを開始します。

<Tabs
  defaultValue="keras"
  values={[
    {label: 'Keras', value: 'keras'},
    {label: 'PyTorch', value: 'pytorch'},
  ]}>
  <TabItem value="keras">

```python
import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback

wandb.init(project="preemptible", resume=True)

if wandb.run.resumed:
    # ベストモデルを復元する
    model = keras.models.load_model(
        wandb.restore("model-best.h5").name
        )
else:
    a = keras.layers.Input(shape=(32,))
    b = keras.layers.Dense(10)(a)
    model = keras.models.Model(input=a, output=b)
モデル.compile("adam", loss="mse")
モデル.fit(np.random.rand(100, 32), np.random.rand(100, 10),
    # 再開時のエポックを設定
    initial_epoch=wandb.run.step, epochs=300,
    # 各エポックで改善された場合、最良のモデルを保存
    callbacks=[WandbCallback(save_model=True, monitor="loss")])
```
  </TabItem>
  <TabItem value="pytorch">T

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_NAME = 'pytorch-resume-run'
CHECKPOINT_PATH = './checkpoint.tar'
N_EPOCHS = 100

# ダミーデータ
X = torch.randn(64, 8, requires_grad=True)
Y = torch.empty(64, 1).random_(2)
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
metric = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epoch = 0
run = wandb.init(project=PROJECT_NAME, resume=True)
if wandb.run.resumed:
    checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
```

```python
model.train()
while epoch < N_EPOCHS:
    optimizer.zero_grad()
    output = model(X)
    loss = metric(output, Y)
    wandb.log({'loss': loss.item()}, step=epoch)
    loss.backward()
    optimizer.step()

    torch.save({ # チェックポイントの場所を保存
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, CHECKPOINT_PATH)
    wandb.save(CHECKPOINT_PATH) # wandbにチェックポイントを保存
    epoch += 1
```
  </TabItem>
</Tabs>

### 再開のガイダンス

W&Bを使ってrunを再開する方法はいくつかありますが、以下で詳しく説明します：

1. [`resume`](./resuming)

   これは、W&Bでrunを再開するためのおすすめの方法です。

   1. 上記で説明したように、runは`wandb.init()`に`resume=True`を渡すことで再開することができます。これは、中断されたrunが終了するところから「自動的に」始めると考えることができます。プロセスが正常に終了しない場合、次に実行するとwandbは最後のステップからログを開始します。
      * 注意：これは、ファイルが `wandb/wandb-resume.json` の場所に保存されているため、失敗したディレクトリと同じディレクトリでスクリプトを実行している場合にのみ機能します。
   2. もう一つの再開方法では、実際のrun idを提供する必要があります： `wandb.init(id=run_id)` そして、再開するとき (もし再開がされていることを確認したい場合は、`wandb.init(id=run_id, resume="must")` を実行します)。
      * `run_id` を管理することで、再開に完全なコントロールを持つことができます。一意のrunに対して一意のidを持つように`run_id`を設定するだけで、 `resume="allow"` を指定して、wandbはそのidのrunを自動的に再開します。これには、 `wandb.util.generate_id()`というidの生成ユーティリティを提供しています。
自動および制御された再開に関する詳細なコンテキストは、[このセクション](resuming.md#undefined)で見つけることができます。
2. [`wandb.restore`](../track/save-restore#examples-of-wandb.restore)
   * これにより、終了した箇所からrunを開始してメトリクスに新しい履歴値をロギングできますが、コードの状態を再確立することはできません。チェックポイントを書いてロードできるようにする必要があります！
   * チェックポイントファイルを介してrunの状態を記録するには、[`wandb save`](../track/save-restore#examples-of-wandb.save)を使用できます。`wandb.save()`でチェックポイントファイルを作成し、`wandb.init(resume=<run-id>)`を使用してロードできます。[こちらのレポート](https://wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W-B--Vmlldzo3MDQ3Mw)では、W&Bを使用したモデルの保存と復元の方法について説明しています。

#### 自動および制御された再開

自動再開は、失敗したプロセスと同じファイルシステム上でプロセスが再開された場合にのみ機能します。ファイルシステムを共有できない場合は、`WANDB_RUN_ID`を設定できます。これは、スクリプトの単一の実行に対応する、プロジェクトごとにグローバルに一意の文字列です。最大64文字までです。すべての非単語文字はダッシュに変換されます。

```python
# 後で再開する際に使用するために、このIDを保存します
id = wandb.util.generate_id()
wandb.init(id=id, resume="allow")
# または環境変数を経由して
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
wandb.init()
```

`WANDB_RESUME`に`"allow"`を設定すると、`WANDB_RUN_ID`を一意の文字列に設定してプロセスの再起動を自動的に処理できます。`WANDB_RESUME`に`"must"`を設定すると、まだ存在しないrunを再開しようとするとwandbがエラーをスローし、新しいrunの自動作成は行われません。

:::caution
複数のプロセスが同じ`run_id`を同時に使用すると、予期しない結果が記録され、レート制限が発生します。
:::

:::info
runを再開し、`wandb.init()`で`notes`が指定されている場合、これらのnotesはUIで追加したnotesに上書きされます。
:::

:::info
スイープの一部として実行されたrunの再開はサポートされていません。
:::
### 事前終了可能なスイープ

もし、スイープエージェントを事前終了が可能な計算環境（例：SLURMジョブ内の事前終了可能なキュー、EC2スポットインスタンス、Google Cloudの事前終了可能なVMなど）で実行している場合、途中で中断されたスイープの実行を自動的に再キューに入れ、完全に実行されるまでリトライされるようにすることができます。

現在の実行が事前終了寸前であることがわかったら、

```
wandb.mark_preempting()
```

を呼び出して、W&Bバックエンドに対して、実行が事前終了されると考えられることをすぐに伝えます。事前終了寸前の状態でマークされた実行がステータスコード0で終了した場合、W&Bはその実行が正常に終了したと見なし、再キューに入れません。事前終了寸前の実行が非ゼロのステータスで終了した場合、W&Bはその実行が事前終了されたものと見なし、スイープに関連付けられた実行キューに自動的に追加します。実行がステータスなしで終了した場合、W&Bは最後のハートビートから5分後に実行を事前終了されたものとしてマークし、スイープの実行キューに追加します。スイープエージェントは、キューが空になるまで実行を消費し続け、その後標準的なスイープ検索アルゴリズムに基づいて新しい実行を生成します。

デフォルトでは、再キューに入れられた実行は最初のステップからログを記録し始めます。中断されたステップでログの記録を再開させるには、`wandb.init(resume=True)`で実行を再開させてください。