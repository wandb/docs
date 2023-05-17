---
description: Add W&B to your Python code script or Jupyter Notebook.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# W＆Bをコードに追加する

<head>
  <title>PythonコードにW＆Bを追加する</title>
</head>

Weights & Biases Python SDKをスクリプトやJupyter Notebookに追加する方法は数多くあります。以下に、W&B Python SDKを自分のコードに統合する「ベストプラクティス」の例を紹介します。

### 元のトレーニングスクリプト

たとえば、Jupyter NotebookのセルやPythonスクリプトに以下のようなコードがあるとします。`main`という関数を定義して、典型的なトレーニングループを模倣しています。各エポックでは、トレーニングデータと検証データセットに対する精度と損失が計算されます。この例では、値はランダムに生成されています。

`config`という辞書を定義し、ハイパーパラメーターの値を格納しています（15行目）。セルの最後で`main`関数を呼び出して、モックトレーニングコードを実行しています。

```python showLineNumbers
#train.py
import random
import numpy as np

def train_one_epoch(epoch, lr, bs): 
  acc = 0.25 + ((epoch/30) +  (random.random()/10))
  loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
  return acc, loss

def evaluate_one_epoch(epoch): 
  acc = 0.1 + ((epoch/20) +  (random.random()/10))
  loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
  return acc, loss

config = {
    'lr' : 0.0001,
    'bs' : 16,
    'epochs': 5
}

def main():
    # `wandb.config`から値を定義することに注意してください。
    # ハードな値を定義する代わりに
    lr = config['lr']
    bs = config['bs']
    epochs = config['epochs']

    for epoch in np.arange(1, epochs):
      train_acc, train_loss = train_one_epoch(epoch, lr, bs)
      val_acc, val_loss = evaluate_one_epoch(epoch)
      
      print('epoch: ', epoch)
      print('training accuracy:', train_acc,'training loss:', train_loss)
      print('validation accuracy:', val_acc,'training loss:', val_loss)

# main関数を呼び出します。       
main()
```

### W&B Python SDKを使ったトレーニングスクリプト

次のコード例は、W&B Python SDKをコードに追加する方法を示しています。CLIでW&Bスイープジョブを開始する場合は、CLIタブを参照してください。JupyterノートブックやPythonスクリプト内でW&Bスイープジョブを開始する場合は、Python SDKタブを参照してください。




<Tabs
  defaultValue="script"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  W&Bスイープを作成するために、コード例に以下の内容を追加しました:

1. 1行目: Wights & Biases Python SDKをインポートします。
2. 6行目: キーと値のペアでスイープ構成を定義するディクショナリオブジェクトを作成します。先の例では、バッチサイズ（`batch_size`）、エポック（`epochs`）、学習率（`lr`）のハイパーパラメータが各スイープで変化します。スイープ構成を作成する方法については、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。
3. 19行目: スイープ構成のディクショナリを[`wandb.sweep`](https://docs.wandb.ai/ref/python/sweep)に渡します。これにより、スイープが初期化されます。これにより、スイープID（`sweep_id`）が返されます。スイープの初期化方法については、[スイープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。
4. 33行目: [`wandb.init()`](https://docs.wandb.ai/ref/python/init) APIを使用して、データの同期とログを行うバックグラウンドプロセスを生成します。[W&B Run](https://docs.wandb.ai/ref/python/run)としてデータを同期・ログするためです。
5. 37-39行目: （オプション）`wandb.config`からの値を、ハードコーディングされた値を定義する代わりに使用します。
6. 45行目: [`wandb.log`](https://docs.wandb.ai/ref/python/log)を使用して、最適化したい指標をログに記録します。設定で定義された指標を記録する必要があります。設定ディクショナリ（この例では`sweep_configuration`）で、`val_acc`値を最大化するようにスイープを定義しました。
7. 54行目: [`wandb.agent`](https://docs.wandb.ai/ref/python/agent) API呼び出しでスイープを開始します。スイープID（19行目）、スイープが実行する関数の名前（`function=main`）、試行する最大ラン数を4（`count=4`）に設定します。W&Bスイープの開始方法については、[スイープエージェントの開始](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)を参照してください。

```python showLineNumbers
import wandb
import numpy as np 
import random

# スイープ構成を定義
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}```
# スイープを初期化する際に設定を渡します。
# （オプション）プロジェクト名を指定することができます。
sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project='my-first-sweep'
  )

# `wandb.config`からハイパーパラメーターの値を取得し、それを使用して
# モデルをトレーニングし、メトリックを返すトレーニング関数を定義します。
def train_one_epoch(epoch, lr, bs): 
  acc = 0.25 + ((epoch/30) +  (random.random()/10))
  loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
  return acc, loss

def evaluate_one_epoch(epoch): 
  acc = 0.1 + ((epoch/20) +  (random.random()/10))
  loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
  return acc, loss

def main():
    run = wandb.init()

    # `wandb.config`から値を定義することに注意してください。
    # 固定値を定義する代わりに
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
      train_acc, train_loss = train_one_epoch(epoch, lr, bs)
      val_acc, val_loss = evaluate_one_epoch(epoch)
wandb.log({
        'epoch': エポック, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      })

# スイープジョブを開始する。
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

  W&Bスイープを作成するには、まずYAML設定ファイルを作成します。設定ファイルには、スイープで探索したいハイパーパラメータを含めます。次の例では、バッチサイズ（`batch_size`）、エポック（`epochs`）、学習率（`lr`）のハイパーパラメータが、各スイープで変更されます。

```yaml
# config.yaml
program: train.py
method: random
name: sweep
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size: 
    values: [16,32,64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```
W&Bスイープ構成の作成方法については、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。

YAMLファイル内の`program`キーには、Pythonスクリプトの名前を指定する必要があります。

次に、コード例に以下を追加します。

1. 1-2行目：Wights & Biases Python SDK（`wandb`）とPyYAML（`yaml`）をインポートします。PyYAMLは、YAML設定ファイルを読み込むために使用されます。
2. 18行目：設定ファイルを読み込みます。
3. 21行目：[`wandb.init()`](https://docs.wandb.ai/ref/python/init) APIを使用して、バックグラウンドプロセスを生成し、データを同期およびログとして[W&B Run](https://docs.wandb.ai/ref/python/run)として記録します。configオブジェクトをconfigパラメータに渡します。
4. 25 - 27行目：ハードコーディングされた値を使用する代わりに、`wandb.config`からハイパーパラメーターの値を定義します。
5. 33-39行目：最適化したい指標を[`wandb.log`](https://docs.wandb.ai/ref/python/log)でログします。設定で定義された指標をログする必要があります。設定ディクショナリ（この例では`sweep_configuration`）内で、`val_acc`の値を最大化するスイープを定義しました。

```python showLineNumbers
import wandb
import yaml
import random
import numpy as np

def train_one_epoch(epoch, lr, bs): 
  acc = 0.25 + ((epoch/30) +  (random.random()/10))
  loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
  return acc, loss

def evaluate_one_epoch(epoch): 
  acc = 0.1 + ((epoch/20) +  (random.random()/10))
  loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
  return acc, loss  

def main():
    # Set up your default hyperparameters
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
run = wandb.init(config=config)

    # ここで `wandb.config`からの値を定義しています
    # 固定値を直接定義するのではなく
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
      train_acc, train_loss = train_one_epoch(epoch, lr, bs)
      val_acc, val_loss = evaluate_one_epoch(epoch)

      wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      })

# メイン関数を呼び出します。
main()
```

CLIに移動します。CLI内で、スイープエージェントが試行する最大ラン数を設定します。この手順は任意です。以下の例では、最大数を5に設定しています。

```bash
NUM=5
```
次に、[`wandb sweep`](https://docs.wandb.ai/ref/cli/wandb-sweep) コマンドでスイープを初期化します。YAMLファイルの名前を指定してください。オプションでプロジェクトの名前をプロジェクトフラグ（`--project`）に指定できます。



```bash

wandb sweep --project sweep-demo-cli config.yaml

```



これにより、スイープIDが返されます。スイープの初期化方法についての詳細は、[Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。



スイープIDをコピーし、続くコードスニペット内の`sweepID`を置き換えて、[`wandb agent`](https://docs.wandb.ai/ref/cli/wandb-agent) コマンドでスイープジョブを開始します。



```bash

wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID

```



スイープジョブの開始方法についての詳細は、[Start sweep jobs](./start-sweep-agents.md)を参照してください。

  </TabItem>

</Tabs>