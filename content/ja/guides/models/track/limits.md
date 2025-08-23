---
title: 実験管理の制限とパフォーマンス
description: W&B のページをより高速かつ快適に保つために、以下の推奨範囲内でログを行ってください。
menu:
  default:
    identifier: ja-guides-models-track-limits
    parent: experiments
weight: 7
---

W&B のページをより高速かつスムーズに利用するために、以下の推奨範囲内でログを記録しましょう。

## ロギングの考慮事項

`wandb.Run.log()` を使って実験のメトリクスを記録します。

### メトリクスの種類数

パフォーマンス向上のため、1つの Project における異なるメトリクスの総数は 10,000 未満に抑えることをおすすめします。

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "a": 1,  # "a" は個別のメトリクス
            "b": {
                "c": "hello",  # "b.c" も個別のメトリクス
                "d": [1, 2, 3],  # "b.d" も個別のメトリクス
            },
        }
    )
```

{{% alert %}}
W&B は自動的にネストされた値をフラットに展開します。辞書を渡した場合、ドット区切りの名前に変換されます。config の場合は3つ、summary の場合は4つまでドットをサポートしています。
{{% /alert %}}



もし Workspace の表示が急に遅くなった場合、直近の Run で意図せず大量の新規メトリクスが記録されていないか確認してください（数千個のプロットに 1〜2個の Run しか表示されていない場合は要注意です）。該当する場合は、その Run を削除し、必要なメトリクスのみで再作成することを検討してください。

### 値のサイズ

1つの値のサイズは 1 MB 未満、1回の `run.log` 呼び出しでの合計サイズは 25 MB 未満にしてください。この制限は `wandb.Media` 型（`wandb.Image` や `wandb.Audio` など）には適用されません。

```python
import wandb

run = wandb.init(project="wide-values")

# おすすめしません
run.log({"wide_key": range(10000000)})

# おすすめしません
with open("large_file.json", "r") as f:
    large_data = json.load(f)
    run.log(large_data)

run.finish()
```

幅広い値（大きなデータ）は、その Run のすべてのメトリクスのプロット表示を遅くさせる場合があります（ワイドな値のメトリクス自身だけでなく）。

{{% alert %}}
推奨されたサイズを超えて値をログしたとしてもデータは保存・追跡されますが、プロットの表示が遅くなる場合があります。
{{% /alert %}}

### メトリクスの記録頻度

記録するメトリクスの性質に合ったロギング頻度を選びましょう。一般的には、サイズが大きい値ほど記録頻度を減らします。W&B の推奨値は以下のとおりです：

- スカラー値: メトリクスあたり <100,000 ポイント
- メディア: メトリクスあたり <50,000 ポイント
- ヒストグラム: メトリクスあたり <10,000 ポイント

```python
import wandb

with wandb.init(project="metric-frequency") as run:
    # おすすめしません
    run.log(
        {
            "scalar": 1,  # 100,000個のスカラー値
            "media": wandb.Image(...),  # 100,000枚の画像
            "histogram": wandb.Histogram(...),  # 100,000個のヒストグラム
        }
    )

    # 推奨例
    run.log(
        {
            "scalar": 1,  # 100,000個のスカラー値
        },
        commit=True,
    )  # まとめてメトリクスを記録

    run.log(
        {
            "media": wandb.Image(...),  # 50,000枚の画像
        },
        commit=False,
    )
    
    run.log(
        {
            "histogram": wandb.Histogram(...),  # 10,000個のヒストグラム
        },
        commit=False,
    )
```



{{% alert %}}
ガイドラインを超えた場合もデータは受け入れられますが、ページの読み込みが遅くなる場合があります。
{{% /alert %}}

### Config のサイズ

run の config の合計サイズは 10 MB 未満にしてください。大きな値を記録すると Project の Workspace や Run テーブル操作が遅くなることがあります。

```python
import wandb 

# 推奨例
with wandb.init(
    project="config-size",
    config={
        "lr": 0.1,
        "batch_size": 32,
        "epochs": 4,
    }
) as run:
    # ここにトレーニングコードを書く
    pass

# おすすめしません
with wandb.init(
    project="config-size",
    config={
        "large_list": list(range(10000000)),  # 大きなリスト
        "large_string": "a" * 10000000,  # 大きな文字列
    }
) as run:
    # ここにトレーニングコードを書く
    pass

# おすすめしません
with open("large_config.json", "r") as f:
    large_config = json.load(f)
    wandb.init(config=large_config)
```

## Workspace の考慮事項 

### Run の数

読み込み速度を維持するため、1 Project あたりの Run の総数は下記を目安にしてください：

- SaaS Cloud：100,000件まで
- 専用クラウド、セルフマネージド：10,000件まで

このしきい値を超えると、Project Workspace や Run テーブルの操作（特に Run のグループ分けや大量の異なるメトリクス集計時）が遅くなります。詳しくは[メトリクスの種類数]({{< relref path="#metric-count" lang="ja" >}})も参照してください。

チームである特定の Run 群（例：最近の Run）のみ頻繁に見る場合は、[使用頻度の低い Run をまとめて新しい「アーカイブ用」Project へ移動]({{< relref path="/guides/models/track/runs/manage-runs.md" lang="ja" >}})し、作業中の Project には最小限の Run のみ残す方法もおすすめです。

### Workspace のパフォーマンス
Workspace の高速化を図るための Tips を紹介します。

#### パネル数
デフォルトでは Workspace は _自動_ モードで、記録した各キーごとに標準パネルを生成します。大規模な Project の Workspace で多数のキーに対応するパネルが含まれると、読み込みや操作が遅くなります。パフォーマンス改善には以下の方法があります：

1. Workspace を手動モードにリセットし、パネルがデフォルトで追加されないようにする
1. [クイック追加]({{< relref path="/guides/models/app/features/panels/#quick-add" lang="ja" >}})を使って、可視化が必要なキーに絞ってパネルを追加する

{{% alert %}}
未使用パネルを1つずつ削除してもパフォーマンスへの影響はごくわずかです。Workspace をリセットし、必要なパネルだけを選んで戻す方法が効果的です。
{{% /alert %}}

Workspace の設定については[パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})を参照してください。

#### セクション数

Workspace に数百ものセクションを作るとパフォーマンスが大幅に落ちます。高レベルなメトリクスのグルーピングでセクションを作り、1メトリクス1セクションのアンチパターンは避けましょう。

セクションが多すぎて遅い場合は、サフィックスではなくプレフィックスでセクションをまとめる設定を使うと、より少ないセクションで高速化が見込めます。

{{< img src="/images/track/section_prefix_toggle.gif" alt="セクション作成の切替え" >}}

### メトリクス数

1 Run あたり 5,000〜100,000 メトリクスを記録する場合は、[手動 Workspace]({{< relref path="/guides/models/app/features/panels/#workspace-modes" lang="ja" >}}) の利用を推奨します。Manual モードでは、表示したいメトリクスに合わせてパネルをまとめて追加・削除できます。必要なプロットに絞ることで表示速度が向上します。なお、プロットされていないメトリクスも通常通り収集・保存されます。

Workspace を手動モードへリセットするには、Workspace のアクション `...` メニューから **Reset workspace** をクリックしてください。リセットしても Run のメトリクスデータへの影響はありません。[Workspace のパネル管理]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})も参照してください。

### ファイル数

1 Run あたりアップロードできるファイル数は 1,000 未満に抑えてください。大量のファイルを記録する場合は W&B Artifacts の活用をおすすめします。1,000 ファイルを超えると Run ページの表示速度が低下します。

### Reports と Workspace の違い

Report は、自由な配置のパネル・テキスト・メディアなどを組み合わせて洞察を共有するためのドキュメント機能です。

一方 Workspace は、数十から数万件のメトリクスを多数の Run にわたり高密度かつ効率的に分析するためのビューです。Reports よりキャッシュ・クエリ・読み込み最適化が行われているため、主に分析用途や 20 件以上のプロットを同時表示したい場合には Workspace の利用をおすすめします。

## Python スクリプトのパフォーマンス

Python スクリプトの性能が低下する主な要因は以下のとおりです：

1. データサイズが大きすぎる場合（学習ループに1 ms以上の遅延が発生することもあります）
2. ネットワーク速度や W&B バックエンドの構成
3. `wandb.Run.log()` を 1 秒間に何度も呼び出している場合（各呼び出し時にわずかな遅延が加わります）

{{% alert %}}
頻繁なロギングで学習が遅くなっていますか？ ロギング戦略を工夫してパフォーマンスを向上させる方法は [こちらの Colab](https://wandb.me/log-hf-colab) をご覧ください。
{{% /alert %}}

W&B 側で課す制限はレート制限のみです。W&B Python SDK は自動で指数的な「バックオフ」と「再試行」を行います。コマンドラインでは「Network failure（ネットワーク障害）」のメッセージが表示されます。無料アカウントの場合、常識的な範囲を超えた利用があれば W&B から連絡がいく場合があります。

## レート制限

W&B SaaS Cloud API では、システムの安定性とユーザー全員の可用性確保のためレート制限が設けられています。これにより、ひとりのユーザーがリソースを独占するのを防ぎます。状況によって低いレート制限になることがあります。

{{% alert %}}
レート制限は今後変更される可能性があります。
{{% /alert %}}

レート制限に達すると、HTTP `429` `Rate limit exceeded` エラーと、[レート制限 HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})がレスポンスに含まれます。

### レート制限 HTTP ヘッダー

以下はレート制限 HTTP ヘッダーの内容です：

| ヘッダー名           | 説明                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------- |
| RateLimit-Limit     | 1ウィンドウで利用可能なクォータ（0〜1000のスケール）                                 |
| RateLimit-Remaining | 現ウィンドウで残っているクォータ（0〜1000のスケール）                                |
| RateLimit-Reset     | クォータがリセットされるまでの残り秒数                                              |

### メトリクスログ API のレート制限

`wandb.Run.log()` はトレーニングデータを W&B へ送ります。この API はオンライン・[オフライン同期]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}})いずれの場合もレート制限があり、合計リクエストサイズやリクエスト回数（指定時間内のリクエスト数）に対する制限があります。

レート制限は Project 単位で適用されます。つまり、チーム内に3つ Project があれば、それぞれ独立した制限枠になります。[有料プラン](https://wandb.ai/site/pricing)のユーザーは無料プランより高い上限が設定されています。

レート制限に達すると、HTTP `429` `Rate limit exceeded` エラーと[レート制限 HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})が返されます。

### メトリクスログ API のレート制限を回避するコツ

レート制限を超えると、`run.finish()` の完了がレートリセットまで遅れる場合があります。回避策として以下を検討してください：

- W&B Python SDK を最新版にアップデートする：W&B Python SDK は定期的に改善されており、効率的な再試行やクォータ最適化の仕組みも進化しています。
- メトリクスの記録頻度を下げる：
  ログの頻度を減らし、クォータを節約しましょう。例として、毎エポックではなく5エポックごとに記録する方法があります。

```python
import wandb
import random

with wandb.init(project="basic-intro") as run:
    for epoch in range(10):
        # 学習と評価のシミュレーション
        accuracy = 1 - 2 ** -epoch - random.random() / epoch
        loss = 2 ** -epoch + random.random() / epoch

        # 5エポックごとにロギング
        if epoch % 5 == 0:
            run.log({"acc": accuracy, "loss": loss})
```

- 手動でデータを同期する：レート制限に達した場合でも Run のデータはローカルに保存されます。コマンド `wandb sync <run-file-path>` でデータを手動同期可能です。詳細は [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}})リファレンスを参照ください。

### GraphQL API のレート制限

W&B Models UI や SDK の [public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) では GraphQL を用いてサーバーへデータの問合せや変更を行います。全ての SaaS Cloud の GraphQL リクエストについて、未認証なら IP アドレス単位、認証済みではユーザー単位でレート制限が設けられています。レート制限は一定時間内のリクエスト数で決まり、ご契約中のプランで上限値が異なります。SDK から Project パスを指定する（一例: Reports, Runs, Artifacts）リクエストは、データベースのクエリ時間による Project 単位の制限も適用されます。

[Teams・Enterprise プラン](https://wandb.ai/site/pricing)利用者は Free プランより高い制限枠を持ちます。W&B Models SDK の public API 利用で制限に達した場合、標準出力にエラーメッセージが表示されます。

レート制限に達した場合は HTTP `429` `Rate limit exceeded` エラーと [レート制限 HTTP ヘッダー]({{< relref path="#rate-limit-http-headers" lang="ja" >}})が返されます。

#### GraphQL API のレート制限回避のコツ

W&B Models SDK の [public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使い大量データを取得する際は、リクエストごとに最低1秒程度の間隔をあけるのをおすすめします。HTTP `429` エラーやレスポンスヘッダーの `RateLimit-Remaining=0` を受け取った場合は、`RateLimit-Reset` の秒数だけ待ってから再試行してください。

## ブラウザ利用時の注意

W&B アプリはメモリ消費が多く、Chrome での利用が最適です。PC のメモリ状況によっては、W&B を3つ以上タブで同時に開くと動作が遅くなることがあります。予想外に動作が遅い場合は、ほかのタブやアプリケーションを閉じることを検討してください。

## パフォーマンス 問題の報告について

W&B ではパフォーマンスを重視しており、負荷や遅延の報告には必ず調査対応しています。調査を迅速に進めるため、読み込みが遅いページがあれば、組み込みのパフォーマンスロガーを実行して主要メトリクスやイベントを記録してください。遅いページのURL末尾に `&PERF_LOGGING` を追加して開き、コンソールの出力内容をアカウント担当者やサポート窓口にご共有ください。

{{< img src="/images/track/adding_perf_logging.gif" alt="PERF_LOGGING の追加" >}}