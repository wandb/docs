---
description: Answers to frequently asked question about W&B Experiments.
displayed_sidebar: default
---

# 実験FAQ

<head>
  <title>実験に関するよくある質問</title>
</head>

以下の質問は、W&Bアーティファクトに関してよくある質問です。

### 1つのスクリプトから複数のrunsを開始する方法は？

`wandb.init` と `run.finish()` を使用して、1つのスクリプトから複数のRunsをログに記録します。

1. `run = wandb.init(reinit=True)`: これを使用して、実行の再初期化を許可します。
2. `run.finish()`: 各runの記録を終了するためにこれを使用します。

```python
import wandb
for x in range(10):
    run = wandb.init(reinit=True)
    for y in range (100):
        wandb.log({"metric": x+y})
    run.finish()
```

あるいは、Pythonのコンテキストマネージャーを使って自動的にログを終了させることもできます。

```python
import wandb
for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x+y})
```
### `InitStartError: wandbプロセスとの通信エラー` <a href="#init-start-error" id="init-start-error"></a>

このエラーは、ライブラリがサーバーにデータを同期するプロセスの起動に問題があることを示しています。

以下の回避策は、特定の環境で問題を解決するのに役立ちます。

<Tabs
  defaultValue="linux"
  values={[
    {label: 'LinuxとOS X', value: 'linux'},
    {label: 'Google Colab', value: 'google_colab'},
  ]}>
  <TabItem value="linux">

```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```
</TabItem>
  <TabItem value="google_colab">

 バージョン`0.13.0`より前の場合、以下の方法をお勧めします：

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>
### wandbをマルチプロセス（例：分散トレーニング）で使う方法は？

トレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()`を実行しなかったプロセスからwandbメソッド呼び出しを避けるようにプログラムを構築する必要があります。\
\
マルチプロセストレーニングを管理するためのいくつかのアプローチがあります:

1. すべてのプロセスで`wandb.init`を呼び出し、共有グループを定義するために[group](../runs/grouping.md) keyword引数を使用します。各プロセスは独自のwandb runを持ち、UIはトレーニングプロセスをまとめて表示します。
2. 1つのプロセスからのみ`wandb.init`を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を介してログに記録するデータを渡します。

:::info
これら2つのアプローチの詳細や、Torch DDPを使用したコード例については、[Distributed Training Guide](./log/distributed-training.md)をご覧ください。
:::

### 人間が読み取れるrun名にプログラムでアクセスする方法は？

[`wandb.Run`](../../ref/python/run.md)の`.name`属性として利用できます。

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

### run名をrun IDに設定することはできますか？

実行名（snowy-owl-10のような）を実行ID（qvlp96vkのような）に上書きしたい場合は、次のスニペットを使用できます。

```python
import wandb
wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```
### ランを名前付けしていません。ランの名前はどこから来ていますか？

ランに明示的に名前をつけない場合、ランをUIで識別しやすくするためにランダムなラン名が割り当てられます。例えば、ランダムなラン名は「pleasant-flower-4」や「misunderstood-glade-2」のようになります。

### ランに関連するgitコミットをどのようにして保存できますか？

スクリプトで`wandb.init`が呼び出されると、リモートリポジトリへのリンクや最新のコミットのSHAを含むgit情報を自動的に保存しようとします。git情報は、[ランページ](../app/pages/run-page.md)に表示されるはずです。表示されない場合は、スクリプトを実行しているシェルの現在の作業ディレクトリが、gitで管理されているフォルダにあるかどうかを確認してください。

gitコミットや実験を実行するために使用されたコマンドは、あなたには見えますが、外部ユーザーには非表示になっています。したがって、プロジェクトが公開されていても、これらの詳細は非公開のままになります。

### オフラインでメトリクスを保存し、後でW&Bに同期することはできますか？

デフォルトでは、`wandb.init`は、リアルタイムでメトリクスをクラウドホストのアプリに同期するプロセスを開始します。マシンがオフラインである場合、インターネットアクセスがない場合、またはアップロードを待機している場合でも、`wandb`をオフラインモードで実行し、後で同期する方法がこちらです。

2つの[環境変数](./environment-variables.md)を設定する必要があります。

1. `WANDB_API_KEY=$KEY`、`$KEY`は[設定ページ](https://app.wandb.ai/settings)から取得したAPIキー
2. `WANDB_MODE="offline"`

以下は、スクリプト内での設定例です:

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

こちらがサンプルのターミナル出力です。

![](/images/experiments/sample_terminal_output.png)

そして準備ができたら、そのフォルダをクラウドに送るために同期コマンドを実行します。

```python
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)

### wandb.initモードの違いは何ですか？

モードは "online"、 "offline"、 "disabled" のいずれかで、デフォルトではonlineです。

`online`(デフォルト)：このモードでは、クライアントはデータをwandbサーバーに送信します。

`offline`：このモードでは、クライアントはデータをwandbサーバーに送信する代わりに、ローカルマシンにデータを保存し、後で [`wandb sync`](https://docs.wandb.ai/ref/cli/wandb-sync?q=sync) コマンドで同期できます。

`disabled`：このモードでは、クライアントはモックされたオブジェクトを返し、すべてのネットワーク通信を防ぎます。クライアントは基本的にno-opのように動作します。つまり、すべてのログが完全に無効になります。ただし、すべてのAPIメソッドのスタブは呼び出し可能です。これは通常、テストで使用されます。

### UI上で実行の状態が "crashed" になっていますが、マシン上ではまだ実行中です。どうすればデータを取り戻せますか？
あなたはおそらくトレーニング中にマシンとの接続が切れたと思われます。[`wandb sync [PATH_TO_RUN]`](https://docs.wandb.ai/ref/cli/wandb-sync)を実行することで、データを回復することができます。実行中のRun IDに対応する`wandb`ディレクトリ内のフォルダが実行パスになります。



### `LaunchError: Permission denied`



エラーメッセージ`Launch Error: Permission denied`が表示される場合、実行を送信しようとしているプロジェクトにログを記録する権限がありません。これはいくつかの理由が考えられます。



1. このマシンでログインしていない。コマンドラインで [`wandb login`](../../ref/cli/wandb-login.md) を実行してください。

2. 存在しないエンティティが設定されています。「Entity」は、あなたのユーザー名または既存のチーム名である必要があります。チームを作成する必要がある場合は、[Subscriptions page](https://app.wandb.ai/billing)にアクセスしてください。

3. プロジェクトの権限がありません。プロジェクトの作成者に、このプロジェクトにrunsをログとして記録できるように、プライバシーを**Open**に設定するよう依頼してください。



### W&Bは`multiprocessing`ライブラリを使用していますか？



はい、W&Bは`multiprocessing`ライブラリを使用しています。例えば、次のようなエラーメッセージが表示される場合があります。



```

現在のプロセスのブートストラップフェーズが終了する前に、新しいプロセスを開始しようとしています。

```



これは、`if name == main`というエントリポイント保護を追加する必要があるかもしれないことを意味します。ただし、スクリプトからW&Bを直接実行しようとしている場合にのみ、このエントリポイント保護を追加する必要があります。