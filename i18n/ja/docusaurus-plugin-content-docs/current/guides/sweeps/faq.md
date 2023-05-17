---
description: Answers to frequently asked question about W&B Sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# FAQ

<head>
  <title>スイープに関するよくある質問</title>
</head>

### W&Bスイープの一部として、すべてのハイパーパラメーターの値を提供する必要がありますか。デフォルト値を設定できますか？

スイープ構成の一部として指定されたハイパーパラメーターの名前と値は、辞書のようなオブジェクトである`wandb.config`でアクセス可能です。

スイープの一部でない`wandb.config`の値は、通常、`wandb.init`の`config`引数に辞書を提供して設定されます。ただし、スイープ中に`wandb.init`に渡される設定情報は、スイープによってオーバーライドされる可能性があるデフォルト値として扱われます。

また、`config.setdefaults`を使用して、意図した振る舞いをより明示的に示すこともできます。両方のメソッドのコードスニペットは以下の通りです：

<Tabs
  defaultValue="wandb.init"
  values={[
    {label: 'wandb.init', value: 'wandb.init'},
    {label: 'config.setdefaults', value: 'config.setdef'},
  ]}>
  <TabItem value="wandb.init">

```python
# ハイパーパラメータのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# デフォルトを提供して実行を開始
#   スイープによって上書き可能
with wandb.init(config=config_default) as run:
    # ここにトレーニングコードを追加してください
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# 実行を開始
with wandb.init() as run:
    # スイープによって設定されていない値を更新
    run.config.setdefaults(config_defaults)
    
    # ここにトレーニングコードを追加してください
```

  </TabItem>
</Tabs>

### SLURMでスイープを実行する方法は？

[SLURMスケジューリングシステム](https://slurm.schedmd.com/documentation.html)を使用してスイープを実行する場合、各スケジュールされたジョブで `wandb agent --count 1 SWEEP_ID` を実行することをお勧めします。これにより、1つのトレーニングジョブを実行してから終了します。これにより、リソースの要求時に実行時間を予測しやすくなり、ハイパーパラメーター検索の並列性を活用できます。
### グリッドサーチを再実行することはできますか？

はい。グリッドサーチを使い果たしたが、W&B Runsの一部を再実行したい場合（例えば、いくつかがクラッシュしたため）があります。再実行したいW&B Runsを削除し、[sweep control page](./sweeps-ui.md)の**Resume**ボタンを選択してください。最後に、新しいSweep IDで新しいW&B Sweepエージェントを開始します。

完了したW&B Runsのパラメータ組み合わせは再実行されません。

### カスタムCLIコマンドをスイープでどのように使用しますか？

コマンドライン引数を渡すことでトレーニングの一部を設定する場合、W&BスイープとカスタムCLIコマンドを使用できます。

例えば、次のコードスニペットは、ユーザーがtrain.pyという名前のPythonスクリプトをトレーニングしているbashターミナルを示しています。ユーザーは、Pythonスクリプト内で解析される値を渡します。

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを使用するには、YAMLファイルの`command`キーを編集します。例えば、上記の例を続けると、次のようになります。

```yaml
program:
  train.py
method: grid
parameters:
  batch_size:
    value: 8
  lr:
    value: 0.0001
command:
  - ${env}
  - python
  - ${program}
  - "-b"
  - your-training-config
  - ${args}
```
`${args}`キーは、スイープ構成ファイル内のすべてのパラメーターに展開され、`argparse`で解析できるように展開されます: `--param1 value1 --param2 value2`

`argparse`で指定しない追加の引数がある場合は、次のように使用できます。

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::info
環境によっては、`python`はPython 2を指すことがあります。Python 3を呼び出すことを確実にするには、コマンドの構成時に`python`の代わりに`python3`を使用してください。

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
:::

### スイープに追加の値を追加する方法はありますか、それとも新しいスイープを開始する必要がありますか？

W&Bスイープが開始されると、スイープ構成を変更することはできません。ただし、任意のテーブルビューに移動し、チェックボックスを使用してrunsを選択し、前のrunsを使用して新しいスイープ構成を作成するために**スイープを作成**メニューオプションを使用できます。

### boolean変数をハイパーパラメーターとしてフラグできますか？

設定のコマンドセクションで`${args_no_boolean_flags}`マクロを使用して、ハイパーパラメータをbooleanフラグとして渡すことができます。これにより、任意のbooleanパラメータがフラグとして自動的に渡されます。`param`が`True`の場合、コマンドは `--param` を受け取ります。`param`が`False`の場合、フラグは省略されます。
### SweepsとSageMakerは使えますか？

はい。一見すると、W&Bを認証し、組み込みのSageMaker estimatorを使っている場合は`requirements.txt`ファイルを作成する必要があります。認証方法や`requirements.txt`ファイルの設定方法については、[SageMaker integration](../integrations/other/sagemaker.md) ガイドをご覧ください。

:::info
完全な例が[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)にありますし、[ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)でも詳しく説明しています。\
また、SageMakerとW&Bを使ったセンチメントアナライザーのデプロイに関する[チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)も読むことができます。\
:::

### W&B Sweepsは、AWS BatchやECSなどのクラウドインフラストラクチャとともに使えますか？

一般的に、候補となるW&Bスイープエージェントが読むことができる場所に`sweep_id`を発行できる方法と、これらのスイープエージェントが`sweep_id`を消費して実行を開始できる方法が必要です。

言い換えれば、`wandb agent`を呼び出す何らかのものが必要です。例えば、EC2インスタンスを立ち上げ、それに対して`wandb agent`を呼び出すことができます。この場合、SQSキューを使って`sweep_id`を複数のEC2インスタンスにブロードキャストし、それらがキューから`sweep_id`を消費して実行を開始するかもしれません。

### スイープがローカルにログを残すディレクトリを変更する方法は？

W&Bが実行データをログに記録するディレクトリのパスを変更するには、環境変数`WANDB_DIR`を設定します。例えば、以下のようになります。

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```

### 複数のメトリクスの最適化

同じ実行で複数のメトリクスを最適化したい場合は、個々のメトリクスの加重和を使用できます。

```python
metric_combined = 0.3*metric_a + 0.2*metric_b + ... + 1.5*metric_n
wandb.log({"metric_combined": metric_combined})
```
新しい組み合わせ指標をログに記録し、最適化目的として設定してください。

```yaml
metric:
  name: metric_combined
  goal: minimize
```

### スイープでのコードログをどのように有効にしますか？

スイープでのコードログを有効にするには、W&B Runを初期化した後に `wandb.log_code()` を追加するだけです。これは、アプリのW&Bプロファイルの設定ページでコードログを有効にしている場合でも必要です。より高度なコードログについては、[こちらの `wandb.log_code()` のドキュメント](https://docs.wandb.ai/ref/python/run#log\_code)を参照してください。

###「Est. Runs」の列とは何ですか？

W&Bは、離散的な検索空間でW&Bスイープを作成するときに発生するであろうRunの推定数を提供します。Runの合計数は、検索空間の直積です。

例えば、次のような検索空間を提供した場合。

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

この例では、直積は9です。W&Bは、W&BアプリのUIで推定されるRun数（**Est. Runs**）としてこの数字を表示します。

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)

W&B SDKを使って推定されるRun数も取得できます。Sweepオブジェクトの `expected_run_count` 属性を使って、推定されたRun数を取得します。

```python
sweep_id = wandb.sweep(sweep_configs, project="your_project_name", entity='your_entity_name')
api = wandb.Api()
sweep=api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```