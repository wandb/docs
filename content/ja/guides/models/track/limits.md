---
title: Experiments の 制限 と パフォーマンス
description: これらの推奨範囲内でログを記録することで、W&B のページをより高速かつ応答性の高い状態に保てます。
menu:
  default:
    identifier: ja-guides-models-track-limits
    parent: experiments
weight: 7
---

W&B のページをより高速かつ応答性良く保つには、以下の推奨範囲内でログを記録してください。

## Logging considerations

`wandb.Run.log()` を使って実験メトリクスを追跡します。

### Distinct metric count

パフォーマンス向上のため、Project 内の一意のメトリクス総数は 10,000 未満に保ってください。

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "a": 1,  # "a" は一意のメトリクス
            "b": {
                "c": "hello",  # "b.c" は一意のメトリクス
                "d": [1, 2, 3],  # "b.d" は一意のメトリクス
            },
        }
    )
```

{{% alert %}}
W&B はネストされた値を自動でフラット化します。つまり、辞書を渡すとドットで区切られた名前に変換されます。config 値の名前に含められるドットは最大 3 個、summary 値では最大 4 個までサポートします。
{{% /alert %}}




Workspace が突然重くなった場合、直近の runs が意図せず数千の新しいメトリクスをログしていないか確認してください。（セクション内に数千のプロットがあるのに 1〜2 個の run しか見えない箇所があれば発見しやすいです。）もしそうなら、その runs を削除し、必要なメトリクスだけで再作成することを検討してください。

### Value width

1 つのログ値のサイズは 1 MB 未満、1 回の `run.log` 呼び出し全体のサイズは 25 MB 未満に制限してください。この制限は `wandb.Image` や `wandb.Audio` などの `wandb.Media` 型には適用されません。

```python
import wandb

run = wandb.init(project="wide-values")

# 非推奨
run.log({"wide_key": range(10000000)})

# 非推奨
with open("large_file.json", "r") as f:
    large_data = json.load(f)
    run.log(large_data)

run.finish()
```

大きな値は、そのメトリクスだけでなく run 内のすべてのメトリクスのプロット読み込み時間に影響を与える可能性があります。

{{% alert %}}
推奨より大きな値をログしてもデータは保存・追跡されますが、プロットの読み込みが遅くなる場合があります。
{{% /alert %}}

### Metric frequency

ログするメトリクスに適した頻度を選びましょう。一般的な目安として、値のサイズが大きいものほど低頻度でログします。W&B の推奨は以下のとおりです。

- Scalars: 各メトリクスあたり <100,000 点
- Media: 各メトリクスあたり <50,000 点
- Histograms: 各メトリクスあたり <10,000 点

```python
import wandb

with wandb.init(project="metric-frequency") as run:
    # 非推奨
    run.log(
        {
            "scalar": 1,  # スカラー 100,000 点
            "media": wandb.Image(...),  # 画像 100,000 点
            "histogram": wandb.Histogram(...),  # ヒストグラム 100,000 点
        }
    )

    # 推奨
    run.log(
        {
            "scalar": 1,  # スカラー 100,000 点
        },
        commit=True,
    )  # バッチ化したステップごとのメトリクスをまとめてコミット

    run.log(
        {
            "media": wandb.Image(...),  # 画像 50,000 点
        },
        commit=False,
    )
    
    run.log(
        {
            "histogram": wandb.Histogram(...),  # ヒストグラム 10,000 点
        },
        commit=False,
    )
```




{{% alert %}}
ガイドラインを超えても W&B はログ済みデータを受け付けますが、ページの読み込みが遅くなる場合があります。
{{% /alert %}}

### Config size

run の config の合計サイズは 10 MB 未満にしてください。大きな値をログすると、Project の Workspaces や runs テーブルの操作が遅くなる可能性があります。

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
    # ここにトレーニング コード
    pass

# 非推奨
with wandb.init(
    project="config-size",
    config={
        "large_list": list(range(10000000)),  # 大きなリスト
        "large_string": "a" * 10000000,  # 大きな文字列
    }
) as run:
    # ここにトレーニング コード
    pass

# 非推奨
with open("large_config.json", "r") as f:
    large_config = json.load(f)
    wandb.init(config=large_config)
```

## Workspace considerations 


### Run count

読み込み時間を短縮するため、1 つの Project 内の runs 総数は次の範囲に収めてください。

- SaaS Cloud では 100,000
- 専用クラウド（Dedicated Cloud）またはセルフマネージドでは 10,000

これらの閾値を超える run 数は、Project Workspaces や runs テーブルに関わる操作を遅くする可能性があります。特に、runs をグループ化したり、run 中に多数の一意メトリクスを収集したりする場合に顕著です。あわせて [メトリクス数]({{< relref path="#metric-count" lang="ja" >}}) のセクションも参照してください。

同じ runs の集合（例: 最近の runs）にチームが頻繁にアクセスする場合は、使用頻度の低い runs をまとめて [別の新しい「archive」Project に移動]({{< relref path="/guides/models/track/runs/manage-runs.md" lang="ja" >}})し、作業用の Project には小さめの集合だけを残すことを検討してください。

### Workspace performance
このセクションでは Workspace のパフォーマンス最適化のためのヒントを紹介します。

#### Panel count
デフォルトでは Workspace は _ 自動 _ で、記録された各キーに対して標準パネルを生成します。大規模な Project の Workspace に多くのキーのパネルが含まれていると、読み込みや操作が遅くなる場合があります。パフォーマンスを改善するには、次を検討してください。

1. Workspace を手動モードにリセットする（デフォルトではパネルを含みません）。
1. 可視化が必要なキーに対してのみ [Quick add]({{< relref path="/guides/models/app/features/panels/#quick-add" lang="ja" >}}) で選択的にパネルを追加する。

{{% alert %}}
未使用のパネルを 1 つずつ削除してもパフォーマンスへの影響はわずかです。代わりに Workspace をリセットし、必要なパネルだけを選択的に追加し直してください。
{{% /alert %}}

Workspace の設定方法については [Panels]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。

#### Section count

Workspace に数百のセクションがあるとパフォーマンスが低下します。メトリクスの高レベルなグルーピングに基づいてセクションを作成し、メトリクスごとに 1 セクションというアンチパターンは避けましょう。

セクションが多すぎてパフォーマンスが低いと感じる場合は、Workspace の設定で「接尾辞」ではなく「接頭辞」でセクションを作成するように切り替えると、セクション数を減らしパフォーマンスが向上することがあります。

{{< img src="/images/track/section_prefix_toggle.gif" alt="セクション作成の切り替え" >}}

### Metric count

1 回の run で 5,000〜100,000 のメトリクスをログする場合は、[手動 Workspace]({{< relref path="/guides/models/app/features/panels/#workspace-modes" lang="ja" >}}) の使用を W&B は推奨します。Manual モードでは、探索したいメトリクスの集合に応じてパネルをまとめて簡単に追加・削除できます。プロットを絞ることで Workspace の読み込みは高速化します。プロットしていないメトリクスも、これまでどおり収集・保存されます。

Workspace を手動モードにリセットするには、Workspace のアクション メニュー `...` をクリックし、**Reset workspace** をクリックします。Workspace をリセットしても run の保存済みメトリクスには影響しません。詳しくは [workspace panel management]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。

### File count

1 回の run でアップロードするファイルの総数は 1,000 未満に保ってください。大量のファイルをログする必要がある場合は W&B Artifacts を使用できます。1 回の run で 1,000 個を超えるファイルは run ページの表示を遅くする可能性があります。

### Reports vs. Workspaces

Report はパネル、テキスト、メディアを自由に組み合わせて構成でき、洞察を同僚と簡単に共有できます。

対照的に、Workspace は数十から数十万の runs にまたがる数十から数千のメトリクスを高密度かつ高パフォーマンスに分析できます。Reports と比べて、Workspace はキャッシュ、クエリ、読み込みが最適化されています。提示用ではなく主に分析で使う Project、または 20 個以上のプロットを同時に表示する必要がある場合には Workspace を推奨します。

## Python script performance

Python スクリプトのパフォーマンスが低下する主な要因は次のとおりです。

1. データ サイズが大きすぎる。大きなデータはトレーニング ループに 1 ms 超のオーバーヘッドを追加する可能性があります。
2. ネットワークの速度や W&B のバックエンド構成
3. 1 秒間に複数回 `wandb.Run.log()` を呼び出している。`wandb.Run.log()` を呼ぶたびにトレーニング ループに小さな遅延が加わるためです。

{{% alert %}}
頻繁なロギングがトレーニング runs を遅くしていますか？ロギング戦略を変更して性能を高める方法については [this Colab](https://wandb.me/log-hf-colab) をチェックしてください。
{{% /alert %}}

W&B はレート制限以外のハードな制限は設けていません。W&B Python SDK は、制限を超えたリクエストに対して指数関数的な「バックオフ」と「リトライ」を自動で行います。コマンドラインには “Network failure” と表示されます。未払いアカウントの場合、常識的な閾値を極端に超える利用が見られた際には W&B から連絡することがあります。

## Rate limits

W&B SaaS Cloud API は、システムの健全性と可用性を保つためにレート制限を実装しています。これは共有インフラストラクチャーで特定のユーザーがリソースを独占するのを防ぎ、すべてのユーザーにサービスが行き渡るようにするための措置です。さまざまな理由で、より低いレート制限が適用される場合があります。

{{% alert %}}
レート制限は予告なく変更されることがあります。
{{% /alert %}}

レート制限に達すると、HTTP `429` `Rate limit exceeded` エラーが返り、レスポンスには[レート制限の HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})が含まれます。

### Rate limit HTTP headers

以下の表はレート制限に関する HTTP ヘッダーを説明します。

| Header name         | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| RateLimit-Limit     | 時間ウィンドウごとの利用可能なクォータ（0〜1000 の範囲でスケーリング）                       |
| RateLimit-Remaining | 現在のレート制限ウィンドウにおける残りクォータ（0〜1000 の範囲でスケーリング）               |
| RateLimit-Reset     | 現在のクォータがリセットされるまでの秒数                                                  |

### Rate limits on metric logging API

`wandb.Run.log()` はトレーニング データを W&B にログします。この API はオンラインまたは[オフライン同期]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}})のいずれかで利用され、いずれの場合もローリング時間ウィンドウに基づくクォータ制限が課されます。これはリクエスト全体のサイズとレート（一定時間あたりのリクエスト数）の両方に対する制限を含みます。

W&B は Project 単位でレート制限を適用します。つまり、チームに 3 つの Projects がある場合、各 Project に独自のレート制限クォータが適用されます。[有料プラン](https://wandb.ai/site/pricing)のユーザーは無料プランよりも高いレート制限が適用されます。

レート制限に達すると、HTTP `429` `Rate limit exceeded` エラーが返り、レスポンスには[レート制限の HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})が含まれます。

### Suggestions for staying under the metrics logging API rate limit

レート制限を超えると、レートがリセットされるまで `run.finish()` が遅延する場合があります。これを避けるため、次の戦略を検討してください。

- W&B Python SDK を最新化する:
  W&B Python SDK は定期的に更新され、リクエストのリトライやクォータ最適化の仕組みが強化されています。常に最新バージョンを使用してください。
- メトリクスのログ頻度を下げる:
  クォータ節約のため、ログ頻度を最小限にします。たとえば、毎エポックではなく 5 エポックごとにログするようコードを変更できます。

```python
import wandb
import random

with wandb.init(project="basic-intro") as run:
    for epoch in range(10):
        # トレーニングと評価をシミュレート
        accuracy = 1 - 2 ** -epoch - random.random() / epoch
        loss = 2 ** -epoch + random.random() / epoch

        # 5 エポックごとにメトリクスをログ
        if epoch % 5 == 0:
            run.log({"acc": accuracy, "loss": loss})
```

- 手動同期:
  レート制限に達した場合、W&B は run データをローカルに保存します。`wandb sync <run-file-path>` コマンドで手動同期できます。詳細は [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) のリファレンスを参照してください。

### Rate limits on GraphQL API

W&B Models の UI と SDK の [Public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) は、データのクエリや変更のためにサーバーへ GraphQL リクエストを送ります。SaaS Cloud のすべての GraphQL リクエストに対して、未認証のリクエストは IP アドレス単位、認証済みのリクエストはユーザー単位でレート制限が適用されます。制限は固定ウィンドウ内のリクエスト レート（1 秒あたりのリクエスト数）に基づき、デフォルト値はご契約プランによって異なります。Project パスを指定する関連 SDK リクエスト（例: Reports、Runs、Artifacts）については、データベース クエリ時間で測定される Project 単位のレート制限が適用されます。

[Teams and Enterprise plans](https://wandb.ai/site/pricing) のユーザーは Free プランより高いレート制限が適用されます。W&B Models SDK の Public API 利用中にレート制限に達した場合、標準出力にエラーを示すメッセージが表示されます。

レート制限に達すると、HTTP `429` `Rate limit exceeded` エラーが返り、レスポンスには[レート制限の HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})が含まれます。

#### Suggestions for staying under the GraphQL API rate limit

W&B Models SDK の [Public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) で大量のデータを取得する場合は、リクエストの間隔を 1 秒以上空けることを検討してください。HTTP `429` `Rate limit exceeded` エラーを受け取った場合、またはレスポンス ヘッダーの `RateLimit-Remaining=0` を確認した場合は、`RateLimit-Reset` に記載の秒数だけ待ってから再試行してください。

## Browser considerations

W&B アプリはメモリ使用量が多くなる場合があり、Chrome で最も良好に動作します。PC のメモリによっては、W&B を同時に 3 つ以上のタブで開いているとパフォーマンスが低下することがあります。予想外に遅いと感じた場合は、他のタブやアプリケーションを閉じることを検討してください。

## Reporting performance issues to W&B

W&B はパフォーマンスを重視しており、遅延の報告はすべて調査します。調査を迅速化するため、読み込みが遅いと感じた際は、主要なメトリクスとパフォーマンス イベントを記録する W&B の組み込みパフォーマンス ロガーを有効化することを検討してください。読み込みが遅いページの URL に `&PERF_LOGGING` パラメータを追加し、コンソール出力をアカウント チームまたはサポートに共有してください。

{{< img src="/images/track/adding_perf_logging.gif" alt="PERF_LOGGING の追加" >}}