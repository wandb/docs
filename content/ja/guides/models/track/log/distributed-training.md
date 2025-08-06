---
title: 分散トレーニング実験をログする
description: W&B を使って、複数の GPU を用いた分散トレーニング実験のログを記録しましょう。
menu:
  default:
    identifier: ja-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

分散トレーニング実験では、複数のマシンやクライアントを並列で使ってモデルをトレーニングします。W&B を使うことで、分散トレーニングの実験管理を簡単にトラッキングできます。ユースケースに応じて、以下のいずれかの方法で分散トレーニング実験を管理できます。

* **単一プロセスをトラッキング**: W&B で rank 0 プロセス（「リーダー」や「コーディネーター」とも呼ばれる）をトラッキングします。これは [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) クラスを用いた分散トレーニング実験のログ取得によく使われる一般的な方法です。
* **複数プロセスをトラッキング**: 複数プロセスの場合は、次のいずれかの方法で対応できます：
   * 各プロセスごとに run を作成して個別にトラッキング。W&B App UI 上でグループ化することも可能です。
   * すべてのプロセスから一つの run にまとめてログを記録。



## 単一プロセスをトラッキング

このセクションでは、rank 0 プロセスで利用できる値やメトリクスをトラッキングする方法を説明します。この方法は、単一のプロセスから利用可能なメトリクスのみを追跡したい場合に適しています。主なメトリクスには GPU/CPU の使用率、共通検証セットでの振る舞い、勾配やパラメータ、代表的なデータ例に関する損失値などがあります。

rank 0 プロセス内で [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を呼び出し、[`wandb.log`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) で実験管理を行います。

以下の [サンプル Python スクリプト（`log-ddp.py`）](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) は、PyTorch DDP を使い一台のマシン上の 2 つの GPU でメトリクスをトラッキングする例です。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)（`torch.nn` の `DistributedDataParallel`）は分散トレーニングで広く使われるライブラリです。基本的な考え方はあらゆる分散トレーニング環境で通用しますが、実装は異なる場合があります。

Python スクリプトの流れ：
1. `torch.distributed.launch` で複数プロセスを起動
1. `--local_rank` コマンドライン引数で rank を判定
1. rank 0 なら [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数内で条件付きで wandb のログを設定

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()

    if args.local_rank == 0:  # メインプロセスのみ
        # wandb run を初期化
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP でモデルを学習
        train(args, run)
    else:
        train(args)
```

[単一プロセスからトラッキングしたメトリクスのダッシュボード例](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)もチェックしてみてください。

このダッシュボードには、2 つの GPU の温度や使用率などシステムメトリクスが表示されます。

{{< img src="/images/track/distributed_training_method1.png" alt="GPU metrics dashboard" >}}

ただし、エポックやバッチサイズごとの損失値（Loss）は単一 GPU からのみログが記録されています。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="Loss function plots" >}}

## 複数プロセスをトラッキング

複数プロセスを W&B でトラッキングする方法は以下の 2 つです。
* [各プロセスごとに個別トラッキング]({{< relref path="distributed-training/#track-each-process-separately" lang="ja" >}})（各プロセスごとに run を作成）
* [全プロセスのログを単一の run にまとめて記録]({{< relref path="distributed-training/#track-all-processes-to-a-single-run" lang="ja" >}})

### 各プロセスごとに個別トラッキング

このセクションでは、各プロセスごとに run を作成して別々にトラッキングする方法を紹介します。各 run ごとにメトリクスやアーティファクトを記録し、トレーニング終了時に `wandb.Run.finish()` をコールして run を終了させることで、すべてのプロセスが正しく終了できます。

複数の Experiments をまたいで run を管理するのが大変と感じる場合は、W&B の初期化時に `group` パラメータに値を渡す（`wandb.init(group='group-name')`）ことで、どの run が同じ実験に属するかを簡単に管理できます。実験内で学習・評価の Run を追跡する方法については [Group Runs]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。

{{% alert %}}
**各プロセスごとのメトリクスを個別にトラッキングしたい場合にこの手法が有効です。** 典型例としては、各ノードごとのデータや予測（データ分布をデバッグするため）、メインノード以外のバッチごとのメトリクスなどが挙げられます。ノード全体のシステムメトリクスや、メインノードで取得できる統計量を得るだけなら、この方法は必須ではありません。
{{% /alert %}}

W&B 初期化時に group パラメータをセットする方法のコード例は以下の通りです:

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()
    # run を初期化
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # すべての run を同じグループにまとめる
    )
    # DDP でモデルを学習
    train(args, run)

    run.finish()  # run を終了
```

W&B App UI 上では、[複数プロセスから記録されたメトリクスの例](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)が確認できます。左サイドバーに 2 つの W&B Runs（グループ化済み）が並んで表示されています。グループをクリックすると、その実験用の専用グループページが表示され、各プロセスごとのメトリクスが独立して確認できます。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="Grouped distributed runs" >}}

上記画像は W&B App UI のダッシュボード例です。サイドバーには 2 つの Experiment が並び、1 つは「null」、もう 1 つ（黄色枠で囲われたもの）は「DPP」となっています。Group ドロップダウンでグループを展開すると、その Experiment に関連付けられた W&B Runs を確認できます。

### 全プロセスを単一の run でトラッキング

{{% alert color="secondary"  %}}
`x_` で始まるパラメータ（例: `x_label`）はパブリックプレビュー機能です。ご意見やご要望は [W&B リポジトリの GitHub issue](https://github.com/wandb/wandb) にご投稿ください。
{{% /alert %}}

{{% alert title="要件" %}}
複数プロセスを単一 run でトラッキングするには、以下が必要です:

- W&B Python SDK バージョン `v0.19.9` 以降
- W&B Server v0.68 以降
{{% /alert  %}}

この方式では、プライマリノードと複数のワーカーノードを利用します。プライマリノード上で W&B run を初期化し、ワーカーノードごとにプライマリノードと同じ run ID を指定して run を初期化します。トレーニング中、各ワーカーノードはプライマリノードと同じ run ID でログを記録します。W&B は全ノードから受け取ったメトリクスを集約し、UI 上でまとめて表示します。

プライマリノード側で [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を呼び出す際、`wandb.Settings` オブジェクトを `settings` パラメータに渡します（例：`wandb.init(settings=wandb.Settings()`)）。設定ポイントは以下です：

1. `mode` パラメータを `"shared"` にして共有モードを有効化
2. [`x_label`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L638) 用の一意なラベルを指定（これにより、どのノードからのデータか UI ログやシステムメトリクスで特定可能。未指定ならホスト名とランダムハッシュで自動生成される）
3. [`x_primary`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L660) を `True` にしてプライマリノードであることを明示
4. オプションで、`x_stats_gpu_device_ids`（例: [0,1,2]）にトラッキング対象 GPU を指定。未指定ならマシン内全 GPU のメトリクスをトラッキング

プライマリノードの run ID は必ず控えておきましょう。ワーカーノードごとにこの run ID を指定して初期化します。

{{% alert %}}
`x_primary=True` でプライマリノードとワーカーノードを区別できます。プライマリノードだけが、ノード間で共有されるファイル（設定ファイルやテレメトリなど）をアップロードします。ワーカーノードはこれらのファイルをアップロードしません。
{{% /alert %}}

各ワーカーノードでは、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を使って下記の項目を指定します:
1. `wandb.Settings` オブジェクトを `settings` パラメータに渡す（`wandb.init(settings=wandb.Settings()`)）
   * `mode` パラメータに `"shared"`
   * 一意な `x_label`
   * `x_primary` を `False` にしてワーカーノードであることを明示
2. プライマリノードと同じ run ID を `id` パラメータに渡す
3. オプションで [`x_update_finish_state`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L772) を `False` に。これにより非プライマリノードが run の状態を早めに `finished` にしないようにし、run の状態が一貫してプライマリノードで管理されます。

{{% alert %}}
run ID を環境変数経由で各ワーカーノードへ渡す運用もおすすめです。
{{% /alert %}}

下記は複数プロセスを単一の run でトラッキングする主な要件のサンプルコードです:

```python
import wandb

# プライマリノードで run を初期化
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="rank_0", 
        mode="shared", 
        x_primary=True,
        x_stats_gpu_device_ids=[0, 1],  # （任意）GPU 0/1 のみトラッキング
        )
)

# プライマリノードの run ID を記録
run_id = run.id

# ワーカーノード上でプライマリの run ID を指定して run を初期化
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared", x_primary=False),
	id=run_id,
)

# 別のワーカーノードの例
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared", x_primary=False),
	id=run_id,
)
```

現実のシナリオでは、各ワーカーノードは別々のマシンで動作している場合もあります。

{{% alert %}}
Kubernetes の GKE 上でマルチノード＆マルチ GPU でモデルを学習させるエンドツーエンド例は [Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) レポートを参照してください。
{{% /alert %}}

複数ノードからのコンソールログを run が属するプロジェクトで確認できます：

1. run を含むプロジェクトに移動
2. 左サイドバーの **Runs** タブをクリック
3. 見たい run をクリック
4. 左サイドバーの **Logs** タブをクリック

UI のコンソールログページ上部の検索バーで `x_label` で指定したラベルごとに絞り込みができます。下記図のように `rank0`, `rank1`, `rank2`, `rank3`, `rank4`, `rank5`, `rank6` といった値を `x_label` に割り当てておけば、個別にログを診断できます。

{{< img src="/images/track/multi_node_console_logs.png" alt="Multi-node console logs" >}}

詳細は [Console logs]({{< relref path="/guides/models/app/console-logs/" lang="ja" >}}) も参照してください。

W&B は全ノードからシステムメトリクスを集約し、共有ダッシュボード上で各ノードのメトリクス（例: `rank_0`, `rank_1`, `rank_2`）がそれぞれ一意のラベルで管理・表示されます。

{{< img src="/images/track/multi_node_system_metrics.png" alt="Multi-node system metrics" >}}

ラインプロットパネルのカスタマイズ方法については [Line plots]({{< relref path="/guides/models/app/features/panels/line-plot/" lang="ja" >}}) をご覧ください。

## ユースケース例

以下のコードスニペットは、高度な分散トレーニングにおけるよくあるシチュエーションを示します。

### プロセスのスポーン

メイン関数内で run をスポーンプロセスに渡す場合は `wandb.setup()` メソッドを使います。

```python
import multiprocessing as mp

def do_work(n):
    with wandb.init(config=dict(n=n)) as run:
        run.log(dict(this=n * n))

def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

### run の共有

run オブジェクトを引数として渡し、プロセス間で run を共有できます。

```python
def do_work(run):
    with wandb.init() as run:
        run.log(dict(this=1))

def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()
    run.finish()  # run を終了


if __name__ == "__main__":
    main()
```

W&B ではログの順序保証はしていません。同期処理はスクリプト作成者側で行ってください。


## トラブルシューティング

W&B で分散トレーニングする際によくある 2 つの問題があります：

1. **学習開始時のハング** - 分散トレーニング用の multiprocessing と wandb の multiprocessing が干渉すると、wandb プロセスがハングすることがあります。
2. **学習終了時のハング** - wandb プロセスが終了条件を把握できないと学習ジョブがハングすることがあります。スクリプトの最後で `wandb.Run.finish()` API を必ず呼び出して、run の終了を W&B 側に通知しましょう。これにより全データのアップロードが完了し、W&B は正しく終了します。

分散ジョブの安定運用のためには `wandb service` コマンドの活用を推奨しています。これらの問題は wandb service が使えないバージョンの SDK で特に発生しやすくなります。

### W&B Service の有効化

ご利用中の W&B SDK バージョンによっては、W&B Service がデフォルトで有効化されています。

#### W&B SDK 0.13.0 以上

W&B SDK バージョン `0.13.0` 以降ではデフォルトで W&B Service が有効です。

#### W&B SDK 0.12.5 以上

W&B SDK バージョン 0.12.5 以上をご利用の場合、Python スクリプトに下記のように `wandb.require` メソッドを追加して service を有効にできます（メイン関数内で利用）：

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # ここにスクリプト本体を書く
```

より快適にご利用いただくため、なるべく最新バージョンへのアップグレードを推奨します。

**W&B SDK 0.12.4 以下の場合**

SDK バージョン 0.12.4 以下では、`WANDB_START_METHOD` 環境変数を `"thread"` に設定してマルチスレッド処理をご利用ください。