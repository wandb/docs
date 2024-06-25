---
description: W&B Experiments に関するよくある質問への回答。
displayed_sidebar: default
---


# Experiments FAQ

<head>
  <title>実験に関するよくある質問</title>
</head>

以下の質問は、W&B Artifacts に関するよくある質問です。

### 1つのスクリプトから複数の Runs を起動するにはどうすればよいですか？

`wandb.init` と `run.finish()` を使用して、1つのスクリプトから複数の Runs をログに記録できます：

1. `run = wandb.init(reinit=True)`: この設定を使用して、Runs を再初期化できるようにします
2. `run.finish()`: Run の最後にこれを使用して、その Run のログ記録を終了します

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

または、自動的にログ記録を終了するPythonのコンテキストマネージャを使用することもできます：

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```

### `InitStartError: wandb プロセスとの通信エラー` <a href="#init-start-error" id="init-start-error"></a>

このエラーは、ライブラリがデータをサーバーに同期するプロセスを起動する際に問題があることを示しています。

以下の回避策は、特定の環境で問題を解決するのに役立ちます：

<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux and OS X', value: 'linux'},
    {label: 'Google Colab', value: 'google_colab'},
  ]}>
  <TabItem value="linux">

```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```
</TabItem>
  <TabItem value="google_colab">

バージョン `0.13.0` より前のものには次の設定をお勧めします：

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>

### マルチプロセッシングを使ったトレーニング、例えば分散トレーニングで wandb を使用するにはどうすればよいですか？

トレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()` を実行していないプロセスから wandb メソッドを呼び出さないようにプログラムを構成する必要があります。

マルチプロセストレーニングを管理するためのいくつかのアプローチがあります：

1. すべてのプロセスで `wandb.init` を呼び出し、共通のグループを定義するために [group](../runs/grouping.md) キーワード引数を使用します。各プロセスにはそれぞれの wandb run があり、UI でトレーニングプロセスがグループ化されます。
2. 1つのプロセスからのみ `wandb.init` を呼び出し、データを [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を介してログに記録します。

:::info
これらの2つのアプローチの詳細および Torch DDP を使用したコード例については、[Distributed Training Guide](./log/distributed-training.md) をご覧ください。
:::

### プログラムで human-readable な run 名にアクセスするにはどうすればよいですか？

[`wandb.Run`](../../ref/python/run.md) の `.name` 属性として利用できます。

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

### run 名を run ID に設定することは可能ですか？

run 名 (例えば snowy-owl-10) を run ID (例えば qvlp96vk) に上書きしたい場合は、次のスニペットを使用できます：

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```

### run に名前を付けませんでした。run 名はどこから来ているのですか？

run に明示的に名前を付けない場合、UI で run を識別しやすいようにランダムな run 名が割り当てられます。例えば、「pleasant-flower-4」や「misunderstood-glade-2」のようなランダムな run 名が表示されます。

### git コミットを run に関連付ける方法は？

スクリプトで `wandb.init` を呼び出すと、最新のコミットのSHAやリモートリポジトリへのリンクなど、git 情報を自動的に検索して保存します。git 情報は [run ページ](../app/pages/run-page.md) に表示されます。表示されない場合は、スクリプト実行時にシェルの現在の作業ディレクトリが git で管理されているフォルダか確認してください。

git コミットと実験の実行に使用されたコマンドは表示されますが、外部ユーザーからは隠されているため、公開プロジェクトの場合でもこれらの詳細は非公開のままです。

### メトリクスをオフラインで保存して後で同期することは可能ですか？

デフォルトでは、`wandb.init` はリアルタイムでメトリクスをクラウドホストされたアプリに同期するプロセスを開始します。マシンがオフラインの場合、インターネットアクセスがない場合、あるいはアップロードを一時的に遅らせたい場合には、以下のように `wandb` をオフラインモードで実行し、後で同期できます。

2つの[環境変数](./environment-variables.md)を設定する必要があります。

1. `WANDB_API_KEY=$KEY`, ここで `$KEY` は [settings page](https://app.wandb.ai/settings) から取得した API キーです
2. `WANDB_MODE="offline"`

これをスクリプトで使用する例はこちらです：

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    wandb.log({"accuracy": i})
```

端末出力のサンプルはこちらです：

![](/images/experiments/sample_terminal_output.png)

準備が整ったら、そのフォルダをクラウドに送信するために sync コマンドを実行するだけです。

```shell
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)

### wandb.init モードの違いは何ですか？

モードは "online"、"offline"、または "disabled" で、デフォルトはオンラインです。

`online`(デフォルト)：このモードでは、クライアントがデータを wandb サーバーに送信します。

`offline`：このモードでは、クライアントはデータをローカルマシンに保存し、後で [`wandb sync`](../../ref/cli/wandb-sync.md) コマンドで同期できます。

`disabled`：このモードでは、クライアントは模擬オブジェクトを返し、すべてのネットワーク通信を防ぎます。クライアントは事実上、操作なしの状態になります。つまり、すべてのログ記録が完全に無効になります。ただし、すべての API メソッドは依然として実行可能です。通常、これはテストで使用されます。

### UI 上で run の状態が "crashed" ですが、マシン上ではまだ実行中です。データを回復するにはどうすればよいですか？

トレーニング中にマシンへの接続が失われた可能性があります。[`wandb sync [PATH_TO_RUN]`](../../ref/cli/wandb-sync.md) を実行してデータを回復できます。run のパスは進行中の run ID に対応する `wandb` ディレクトリ内のフォルダです。

### `LaunchError: Permission denied`

エラーメッセージ `Launch Error: Permission denied` が表示される場合、ログを送信しようとしているプロジェクトに対する権限がありません。考えられる理由はいくつかあります。

1. このマシンにログインしていない。コマンドラインで [`wandb login`](../../ref/cli/wandb-login.md) を実行します。
2. 存在しないエンティティを設定した。"Entity" はユーザー名か既存のチームの名前である必要があります。チームを作成する必要がある場合は、[Subscriptions page](https://app.wandb.ai/billing) にアクセスしてください。
3. プロジェクト権限がありません。プロジェクトの作成者にプライバシーを **Open** に設定してもらい、ログをこのプロジェクトに送信できるようにしてください。

### W&B は `multiprocessing` ライブラリを使用していますか？

はい、W&B は `multiprocessing` ライブラリを使用しています。例えば次のようなエラーメッセージが表示される場合があります：

```
現在のプロセスのブートストラップフェーズが終了する前に、新しいプロセスを開始しようとしました。
```

この場合、エントリポイント保護 `if name == main` を追加する必要があるかもしれません。これは W&B をスクリプトから直接実行しようとしている場合にのみ必要です。