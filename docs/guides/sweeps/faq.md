---
description: W&B Sweeps に関するよくある質問への回答。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# FAQ

<head>
  <title> Sweeps に関するよくある質問</title>
</head>

### W&B Sweep の一部としてすべてのハイパーパラメータの値を提供する必要がありますか？デフォルト値を設定できますか？

sweep 設定の一部として指定されたハイパーパラメータの名前と値は、`wandb.config` という辞書のようなオブジェクトでアクセスできます。

sweep の一部でない run では、通常 `wandb.init` の `config` 引数に辞書を渡して `wandb.config` の値を設定します。しかし、sweep 中では、`wandb.init` に渡された設定情報はデフォルト値として扱われ、sweep によって上書きされる可能性があります。

`config.setdefaults` を使用して、意図した振る舞いを明確にすることもできます。以下のコードスニペットで両方のメソッドを示します：

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

# run を開始してデフォルト値を提供
#   sweep によって上書きされる可能性のあるデフォルト値
with wandb.init(config=config_default) as run:
    # ここにトレーニングコードを追加
    ...
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# ハイパーパラメータのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始
with wandb.init() as run:
    # sweep によって設定されなかった値を更新
    run.config.setdefaults(config_defaults)

    # ここにトレーニングコードを追加
```

  </TabItem>
</Tabs>

### SLURM で sweeps をどのように実行すべきですか？

[SLURM スケジューリングシステム](https://slurm.schedmd.com/documentation.html) で sweeps を使用する際には、各スケジュール済ジョブで `wandb agent --count 1 SWEEP_ID` を実行することをお勧めします。これにより、単一のトレーニングジョブが実行されて終了します。これにより、リソース要求時に実行時間を予測しやすくなり、ハイパーパラメータ検索の並列処理を活用できます。

### グリッド検索を再実行できますか？

はい。グリッド検索を使い切った後でも、いくつかの W&B Runs を再実行したい場合（例えば、いくつかがクラッシュした場合）、再実行したい W&B Runs を削除し、その後 [sweep コントロールページ](./sweeps-ui.md) の **Resume** ボタンを選択します。最後に、新しい Sweep ID で新しい W&B Sweep エージェントを開始します。

完了した W&B Runs のパラメータの組み合わせは再実行されません。

### カスタムCLIコマンドで sweeps を使用する方法

通常、トレーニングの一部をコマンドライン引数として渡す場合、W&B Sweeps とカスタム CLI コマンドを使用できます。

例えば、以下のコードスニペットは、ユーザーが train.py という名前の Python スクリプトをトレーニングしている bash ターミナルを示しています。ユーザーは値を渡し、それが Python スクリプト内で解析されます：

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

カスタムコマンドを使用するには、YAML ファイルの `command` キーを編集します。例えば、上記の例を続けると次のようになります：

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

`${args}` キーは sweep 設定ファイル内のすべてのパラメータに展開され、それらが `argparse` によって解析できるように展開されます：`--param1 value1 --param2 value2`

追加の引数を `argparse` で特定したくない場合は、次のようにします：

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::情報
環境に応じて、`python` は Python 2 を指すことがあります。したがって、コマンドを設定する際には `python` の代わりに `python3` を使用して Python 3 を呼び出すようにします：

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

### sweep に追加の値を追加する方法、または新しいものを開始する必要がありますか？

W&B Sweep が開始されると、Sweep 設定を変更することはできません。しかし、任意のテーブルビューに移動し、チェックボックスを使用して run を選択してから、**Create sweep** メニューオプションを使用して以前の run を使用して新しい sweep 設定を作成できます。

### ブール変数をハイパーパラメータとしてフラグできますか？

config の command セクションで `${args_no_boolean_flags}` マクロを使用して、ハイパーパラメータとしてブールフラグを渡すことができます。これにより、ブールパラメータが自動的にフラグとして渡されます。`param` が `True` の場合、コマンドは `--param` を受け取り、`param` が `False` の場合、フラグは省略されます。

### Sweeps と SageMaker を使用できますか？

はい、一目で分かるように、W&B を認証し、組み込みの SageMaker 推定器を使用する場合は `requirements.txt` ファイルを作成する必要があります。認証方法や `requirements.txt` ファイルの設定については、[SageMaker integration](../integrations/other/sagemaker.md) ガイドをご覧ください。

::: 情報
完全な例は [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) で利用可能です。また、当社の [ブログ](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) でも詳しく読むことができます。\
また、SageMaker と W&B を使用した感情分析器のデプロイに関する [チュートリアル](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) もご覧いただけます。
:::

### W&B Sweeps を AWS Batch、ECS などのクラウドインフラストラクチャで使用できますか？

一般的に、`sweep_id` を潜在的な W&B Sweep エージェントが読み取れる場所に公開し、これらの Sweep エージェントが `sweep_id` を消費して実行を開始できる方法が必要です。

つまり、`wandb agent` を呼び出すものが必要です。例えば、EC2 インスタンスを立ち上げて `wandb agent` を呼び出します。この場合、SQS キューを使用して `sweep_id` を複数の EC2 インスタンスにブロードキャストし、それらがキューから `sweep_id` を消費して実行を開始することが考えられます。

### sweep のログをローカルに記録するディレクトリを変更する方法

実行データを記録するディレクトリのパスを変更するには、環境変数 `WANDB_DIR` を設定します。例えば：

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```

### 複数のメトリクスを最適化

同じ run で複数のメトリクスを最適化する場合、個々のメトリクスの加重和を使用できます。

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

新しい結合メトリクスをログに記録し、それを最適化目標として設定してください：

```yaml
metric:
  name: metric_combined
  goal: minimize
```

### Sweeps でコードを記録する方法

sweeps でコードを記録するには、W&B Run を初期化した後に単純に `wandb.log_code()` を追加します。これは、W&B プロフィールの設定ページでコード記録を有効にしている場合でも必要です。より高度なコード記録については、[ここで `wandb.log_code()` のドキュメント](../../ref/python/run.md#log_code)を参照してください。

### "Est. Runs" カラムとは何ですか？

W&B は、離散的な探索空間を持つ W&B Sweep を作成する際に発生する見積もり Run 数を提供します。Run の総数は探索空間のデカルト積です。

例えば、次の探索空間を提供するとしましょう：

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

この例のデカルト積は 9 です。W&B は、この数値を W&B アプリの UI に推定実行数 (**Est. Runs**) として表示します：

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)

W&B SDK を使用して推定 Run 数を取得することもできます。Sweep オブジェクトの `expected_run_count` 属性を使用して推定 Run 数を取得します：

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
