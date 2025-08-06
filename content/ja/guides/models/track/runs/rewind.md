---
title: run を巻き戻す
description: 巻き戻し
menu:
  default:
    identifier: rewind
    parent: what-are-runs
---

# run をリワインドする
{{% alert color="secondary" %}}
run のリワインド機能はプライベートプレビュー段階です。この機能の利用をご希望の場合は、support@wandb.com まで W&B サポートにご連絡ください。

W&B では現在、以下はサポートされていません:
* **ログのリワインド**: 新しい run セグメントではログがリセットされます。
* **システムメトリクスのリワインド**: リワインドポイント以降の新しいシステムメトリクスのみが W&B に記録されます。
* **Artifact の関連付け**: Artifact は、それを作成したソース run に関連付けられます。
{{% /alert %}}

{{% alert %}}
* run をリワインドするには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョンが `0.17.1` 以上である必要があります。
* 単調増加となる step のみ使用可能です。[`define_metric()`]({{< relref "/ref/python/sdk/classes/run#define_metric" >}}) で定義された非単調な step では動作しません。これは run の履歴やシステムメトリクスの時間順序が損なわれるためです。
{{% /alert %}}

run の履歴を修正または調整したいが、元データを失いたくない場合に、run をリワインドできます。さらに、run をリワインドした時点から新しいデータを記録することが可能です。W&B はリワインドした run の summary メトリクスを、新たに記録された履歴から再計算します。これにより以下のような振る舞いとなります:
- **履歴の切り捨て**: W&B は履歴をリワインドポイントまで切り捨て、新たなデータ記録を可能にします。
- **summary メトリクス**: 新しくログした履歴に基づき再計算されます。
- **設定の保持**: 元の設定が保持され、新しく設定をマージすることも可能です。



run をリワインドすると、その run の状態が指定した step まで巻き戻され、元データは保護されたまま一貫した run ID を維持します。つまり:

- **run のアーカイブ**: 元の run はアーカイブされます。run には [Run Overview]({{< relref "./#overview-tab" >}}) タブからアクセス可能です。
- **Artifact の関連付け**: Artifact はそれを作成した run に関連付けられます。
- **イミュータブルな run ID**: 正確な状態からのフォークを一貫させるために導入されています。
- **イミュータブル run ID のコピー**: run 管理を改善するため、イミュータブルな run ID をコピーできるボタンが追加されています。

{{% alert title="リワインドとフォークの互換性" %}}
フォークはリワインドを補完します。

run からフォークする際は、特定の時点から run の新しいブランチを作成し、異なるパラメータやモデルで試すことができます。

run をリワインドする場合、run の履歴自体を修正・調整することが可能です。
{{% /alert %}}



## run をリワインドする

`wandb.init()` の `resume_from` 引数で run の履歴を特定の step まで「リワインド」できます。run の名前とリワインドしたい step を指定してください:

```python
import wandb
import math

# 最初の run を初期化してメトリクスを記録
# your_project_name と your_entity_name を書き換えてください！
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run を特定の step からリワインドし、step 200 からメトリクスの記録を再開
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しい run でロギングを継続
# 最初のいくつかの step では run1 の値をそのまま記録
# step 250 以降はスパイキーなパターンを記録
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # run1 の続きとしてスパイクなしでロギング
    else:
        # step 250 からスパイキーな振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微細なスパイクパターンを適用
        run2.log({"metric": subtle_spike, "step": i})
    # すべての step で追加メトリクスも記録
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされた run を表示する


run をリワインドした後は、W&B App の UI からアーカイブ済みの run を確認できます。アーカイブされた run を表示するには、以下の手順に従ってください。

1. **Overview タブにアクセス:** run ページの [**Overview** タブ]({{< relref "./#overview-tab" >}}) に移動します。このタブは run の詳細と履歴を包括的に表示します。
2. **Forked From フィールドを確認:** **Overview** タブ内で `Forked From` フィールドを探します。このフィールドには再開履歴が記録されています。**Forked From** フィールドにはソース run へのリンクも含まれており、元の run にさかのぼってリワインドの履歴全体をたどることができます。

`Forked From` フィールドを使えば、アーカイブされた再開のツリーを簡単にたどって、それぞれのリワインドの順序や起点を把握できます。 

## リワインドした run からフォークする

リワインド後の run からフォークするには、`wandb.init()` の [`fork_from`]({{< relref "/guides/models/track/runs/forking" >}}) 引数にソース run ID とフォークしたい step を指定します。

```python 
import wandb

# 特定の step から run をフォークする
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しい run でロギングを継続
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```