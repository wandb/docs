---
title: 実験管理の制限とパフォーマンス
description: W&B のページをより高速かつ快適に利用するために、以下の推奨範囲内でログを行ってください。
menu:
  default:
    identifier: limits
    parent: experiments
weight: 7
---

W&B のページをより高速かつ快適に保つために、以下の推奨範囲内でログを取得することをおすすめします。

## ロギングに関する考慮事項

実験のメトリクスをトラッキングするには `wandb.Run.log()` を使用します。

### メトリクスの種類の数

パフォーマンスを維持するために、1つの Project 内で利用するメトリクス（種類）の合計は 10,000 未満に抑えてください。

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "a": 1,  # "a" は異なるメトリクス
            "b": {
                "c": "hello",  # "b.c" は異なるメトリクス
                "d": [1, 2, 3],  # "b.d" も異なるメトリクス
            },
        }
    )
```

{{% alert %}}
W&B はネストされた値を自動でフラット化します。つまり、辞書を渡すとドット区切りの名前に変換されます。config の値には3階層まで、summary の値には4階層までのドットをサポートしています。
{{% /alert %}}

ワークスペースが突然遅くなった場合は、最近の run で数千もの新しいメトリクスが意図せずロギングされていないか確認してください。（この状況は、1〜2個の run だけが表示されているグラフが数千個並んでいるセクションを見つけることで気づくことが多いです。）もしそうであれば、その run を削除し、必要なメトリクスで再作成することを検討してください。

### 値の幅（バリューサイズ）

1 回のログで記録する値のサイズは 1 MB 未満、1 回の `run.log` 呼び出し全体では 25 MB 未満にしてください。この制限は `wandb.Media` 系（`wandb.Image`、`wandb.Audio` など）には適用されません。

```python
import wandb

run = wandb.init(project="wide-values")

# 推奨されません
run.log({"wide_key": range(10000000)})

# 推奨されません
with open("large_file.json", "r") as f:
    large_data = json.load(f)
    run.log(large_data)

run.finish()
```

大きな値を記録すると、そのメトリクスだけでなく、run 内のすべてのメトリクスのグラフ表示に影響し、読み込み速度が遅くなる場合があります。

{{% alert %}}
推奨値より大きな値をログに記録しても、データそのものは保存・追跡されます。ただし、グラフの読み込みが遅くなる可能性があります。
{{% /alert %}}

### メトリクスの記録頻度

ロギングするメトリクスに応じて適切な記録頻度を選びましょう。一般的な目安として、情報量が多い値ほど記録頻度を下げるのがおすすめです。W&B の推奨値は以下の通りです：

- スカラー値: 1 メトリクスあたり <100,000 ポイント
- メディア: 1 メトリクスあたり <50,000 ポイント
- ヒストグラム: 1 メトリクスあたり <10,000 ポイント

```python
import wandb

with wandb.init(project="metric-frequency") as run:
    # 推奨されません
    run.log(
        {
            "scalar": 1,  # 100,000 スカラー
            "media": wandb.Image(...),  # 100,000 画像
            "histogram": wandb.Histogram(...),  # 100,000 ヒストグラム
        }
    )

    # 推奨
    run.log(
        {
            "scalar": 1,  # 100,000 スカラー
        },
        commit=True,
    )  # 複数ステップ分まとめてコミット

    run.log(
        {
            "media": wandb.Image(...),  # 50,000 画像
        },
        commit=False,
    )
    
    run.log(
        {
            "histogram": wandb.Histogram(...),  # 10,000 ヒストグラム
        },
        commit=False,
    )
```

{{% alert %}}
ガイドラインを超えてデータを記録した場合でも、W&B はログを受け付けます。ただしページの読み込みが遅くなる場合があります。
{{% /alert %}}

### config のサイズ

run の config 全体サイズは 10 MB 未満に抑えてください。大きな値をログに記録すると、Project のワークスペースや runs テーブルの操作が遅くなる可能性があります。

```python
import wandb 

# 推奨
with wandb.init(
    project="config-size",
    config={
        "lr": 0.1,
        "batch_size": 32,
        "epochs": 4,
    }
) as run:
    # ここにトレーニングコード
    pass

# 推奨されません
with wandb.init(
    project="config-size",
    config={
        "large_list": list(range(10000000)),  # 大きなリスト
        "large_string": "a" * 10000000,  # 大きな文字列
    }
) as run:
    # ここにトレーニングコード
    pass

# 推奨されません
with open("large_config.json", "r") as f:
    large_config = json.load(f)
    wandb.init(config=large_config)
```

## Workspace に関する考慮事項

### run の数

読み込み速度を維持するため、1つの Project あたりの run 数の上限を次のようにしてください：

- SaaS Cloud: 100,000
- 専用クラウドまたはセルフマネージド: 10,000

この閾値を超えると、Project のワークスペースや runs テーブルを操作する際（特に run をグループ化したり、多数のメトリクスを集計している場合）に速度低下が発生します。詳しくは [Metric count]({{< relref "#metric-count" >}}) もご覧ください。

チームでよく使う特定の run のセット（たとえば最近の run群など）がある場合、[あまり使わない run をまとめて]({{< relref "/guides/models/track/runs/manage-runs.md" >}}) "archive" 用 Project に移動し、作業用 Project をコンパクトに保つのも有効です。

### Workspace パフォーマンス
このセクションでは、ワークスペースのパフォーマンスを最適化するためのヒントを紹介します。

#### パネル数
デフォルトでは Workspace は「自動」となっていて、ログに記録した各キーの標準パネルが自動生成されます。大規模プロジェクトで多数のキーのパネルが表示されていると、ワークスペースの読み込みや利用が遅くなる場合があります。パフォーマンス改善のために、以下を検討してください：

1. ワークスペースを「手動モード（manual）」にリセットし、パネルが表示されない状態にする
1. 必要なキーにだけ、[クイック追加]({{< relref "/guides/models/app/features/panels/#quick-add" >}}) で必要なパネルのみ追加する

{{% alert %}}
使わないパネルを1つずつ削除してもパフォーマンス向上にはあまり効果がありません。それよりも、ワークスペースをリセットした上で必要なパネルだけを選んで追加するのが有効です。
{{% /alert %}}

ワークスペースの設定に関して詳しく知りたい方は、[Panels]({{< relref "/guides/models/app/features/panels/" >}}) もご覧ください。

#### セクション数

ワークスペースにセクションが何百もあるとパフォーマンスが低下します。高次元のメトリクスのグルーピング単位でセクションを作成し、逆に「1メトリクスごとに1セクション」型の過度な細分化は避けましょう。

もしセクション数が多くて動作が遅い場合は、セクション作成を末尾ではなく接頭語（プレフィックス）ごとにまとめる設定を使うことで、セクション数を減らしてパフォーマンス向上が期待できます。

{{< img src="/images/track/section_prefix_toggle.gif" alt="Toggling section creation" >}}

### メトリクス数

1 run あたり5,000〜100,000 メトリクスを記録する場合、[ワークスペースを手動モード]({{< relref "/guides/models/app/features/panels/#workspace-modes" >}}) に変更することをおすすめします。手動モードでは、興味のある複数のメトリクスセットに応じて、パネルをまとめて追加・削除できます。フォーカスしたグラフだけ表示されることで、ワークスペース全体の読み込みも高速です。グラフ表示しないメトリクスも、記録・保存は通常通り行われます。

ワークスペースを手動モードにリセットするには、ワークスペースアクションの `...` メニューをクリックし、**Reset workspace** を押してください。ワークスペースのリセットは、run のメトリクスの保存には影響しません。詳しくは [workspace panel management]({{< relref "/guides/models/app/features/panels/" >}}) を参照してください。

### ファイル数

1 run でアップロードするファイルの総数は 1,000 個未満に保つようにしましょう。大量のファイルを記録したい場合は、W&B Artifacts の利用をご検討ください。1 run で1,000個を超えると run ページの表示が遅くなります。

### Reports と Workspaces の違い

Report は、自由なパネル・テキスト・メディアを組み合わせて構成し、同僚とインサイトを簡単に共有できるフリーフォームのレポートです。

一方、Workspace は何十〜何千ものメトリクス・何百〜何十万 run に渡る高密度で高速な分析に最適化されています。Workspace は Reports よりもキャッシュやクエリ、ロードの効率が高く、分析利用時や20個以上のグラフを同時に表示する場合におすすめです。

## Python スクリプトのパフォーマンス

Python スクリプトのパフォーマンスが低下する主な要因は以下です：

1. データサイズが大きすぎる（巨大なデータはトレーニングループに 1ms 超のオーバーヘッドが入る場合があります）
2. ネットワーク速度や W&B バックエンドの設定
3. `wandb.Run.log()` を1秒間に何度も呼び出すケース（呼ぶたびにトレーニングループへ僅かな待ち時間が加わるため）

{{% alert %}}
頻繁なログがトレーニング run の速度低下を招いていませんか？より良いパフォーマンスを得るための方法は [こちらの Colab](https://wandb.me/log-hf-colab) をご覧ください。
{{% /alert %}}

W&B はレートリミット（利用制限）以外の上限値は明示していません。W&B Python SDK は自動的に指数バックオフ＆再試行を行い、許容値を超えたリクエストに対応します。利用制限超過時はコマンドライン上で「Network failure」と表示します。無料アカウントの場合、著しく高い利用頻度が続く場合は W&B からご連絡する場合があります。

## レートリミット

W&B SaaS Cloud の API にはレートリミット（利用制限）が設けられており、システム全体の安定性と可用性の確保に役立っています。これにより、1ユーザーが共有インフラでリソースを独占するのを防ぎ、サービス全体のアクセシビリティを確保します。レートリミット値はさまざまな要因で変更されることがあります。

{{% alert %}}
レートリミット値は変更される場合があります。
{{% /alert %}}

レートリミット超過時は、HTTP `429` `Rate limit exceeded` エラーと [レートリミット HTTP ヘッダー]({{< relref "#rate-limit-http-headers" >}}) が返されます。

### レートリミット HTTP ヘッダー

下表はレートリミット HTTP ヘッダーの説明です：

| ヘッダー名             | 説明                                                                          |
| ---------------------  | --------------------------------------------------------------------------  |
| RateLimit-Limit        | 1つの時間枠内で利用可能なクォータ（0〜1000の範囲でスケール）                 |
| RateLimit-Remaining    | 現在の時間枠で残っているクォータ（0〜1000の範囲でスケール）                  |
| RateLimit-Reset        | 現在のクォータがリセットされるまでの残り秒数                                  |

### メトリクスロギング API のレートリミット

`wandb.Run.log()` でトレーニングデータを W&B へ送信します。この API はオンラインまたは [オフライン同期]({{< relref "/ref/cli/wandb-sync.md" >}}) のいずれでも利用でき、いずれの場合も一定時間内のリクエスト数・リクエストサイズに対する制限があります。

W&B では Project ごとにレートリミットが適用されます。たとえばチーム内に 3 つ Project があれば、それぞれ独立したクォータが割り当てられます。[有料プラン](https://wandb.ai/site/pricing) のユーザーは無料プランよりも高いリミットが利用できます。

リミット超過時は、HTTP `429` `Rate limit exceeded` エラーおよび [レートリミット HTTP ヘッダー]({{< relref "#rate-limit-http-headers" >}}) が返されます。

### メトリクスロギング API のレートリミットを回避するためのヒント

リミット超過時は `run.finish()` の終了がリミット解除まで遅れる場合があります。下記の方法をご検討ください:

- W&B Python SDK のバージョンを更新する: 最新版 SDK を利用してください。SDK にはリトライ戦略や使用量最適化機構がアップデートされています。
- メトリクスの記録頻度を減らす: クォータを節約するため、たとえば「毎エポック」ではなく「5エポックごと」に記録するなど、頻度調整をしましょう。

```python
import wandb
import random

with wandb.init(project="basic-intro") as run:
    for epoch in range(10):
        # トレーニング＆評価をシミュレート
        accuracy = 1 - 2 ** -epoch - random.random() / epoch
        loss = 2 ** -epoch + random.random() / epoch

        # 5 エポックごとにログ
        if epoch % 5 == 0:
            run.log({"acc": accuracy, "loss": loss})
```

- データを手動同期する: リミット超過中は run データがローカルに保存されます。`wandb sync <run-file-path>` コマンドで手動同期可能です。詳細は [`wandb sync`]({{< relref "/ref/cli/wandb-sync.md" >}}) リファレンスを参照してください。

### GraphQL API のレートリミット

W&B Models の UI や SDK の [公開API]({{< relref "/ref/python/public-api/api.md" >}}) では、GraphQL 経由でデータの問い合わせや編集を行います。SaaS Cloud では未認証リクエストは IP アドレスごと、認証済みリクエストはユーザーごとにレートリミットがかかります。リクエスト数（毎秒）ベースでプランによりリミット値が異なります。また、Project を指定するリクエストでは、Project ごとにDBクエリ時間にもとづいた制限も適用されます。

[Teams および Enterprise プラン](https://wandb.ai/site/pricing) のユーザーは、Free プランよりも高いリミット値となっています。
W&B Models SDK の公開 API 利用時にリミット超過に達すると、標準出力に原因メッセージが表示されます。

リミット超過時は、HTTP `429` `Rate limit exceeded` エラーおよび [レートリミット HTTP ヘッダー]({{< relref "#rate-limit-http-headers" >}}) が返されます。

#### GraphQL API のレートリミット回避のヒント

W&B Models SDK の [公開API]({{< relref "/ref/python/public-api/api.md" >}}) で大量のデータ取得をする場合、各リクエスト間に最低1秒の間隔をあけることを推奨します。HTTP `429` エラーや `RateLimit-Remaining=0` を受け取った場合は、`RateLimit-Reset` で指定された秒数だけ待ってからリトライしてください。

## ブラウザ利用時の注意

W&B はメモリを多く使用する場合があり、Chrome での利用が最もパフォーマンス良好です。お使いのマシンのメモリ量によっては、W&B を3タブ以上開いていると速度低下が発生する場合があります。予想以上に動作が遅い場合、他のタブやアプリケーションを閉じてみてください。

## パフォーマンス問題の報告について

W&B ではパフォーマンスを重要視しており、ラグのご報告1件ごとに調査を行います。できるだけ早く調査を進めるため、ページの読み込みが遅い場合は W&B のビルトイン「パフォーマンスロガー」を活用し、主要なメトリクスやパフォーマンスイベントを記録しましょう。遅いページのURLに `&PERF_LOGGING` パラメータを追加し、コンソール出力をアカウント担当者やサポートへ共有してください。

{{< img src="/images/track/adding_perf_logging.gif" alt="Adding PERF_LOGGING" >}}