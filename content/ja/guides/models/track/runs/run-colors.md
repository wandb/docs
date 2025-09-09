---
title: run の色をカスタマイズ
menu:
  default:
    identifier: ja-guides-models-track-runs-run-colors
    parent: Customize run colors
---

W&B は、あなたの project で作成する各 run に自動で色を割り当てます。テーブルやグラフで他の run と視覚的に区別しやすくするために、run のデフォルトの色を変更できます。project の Workspace をリセットすると、テーブル内のすべての run のデフォルトの色が復元されます。

run の色はローカルスコープです。project ページでは、カスタム色は自分の Workspace にのみ適用されます。Reports では、run のカスタム色はセクション単位でのみ適用されます。同じ run を複数のセクションで可視化でき、セクションごとに異なるカスタム色を使えます。

## デフォルトの run の色を編集

1. project サイドバーから **Runs** タブをクリックします。
2. **Name** 列で run 名の隣にあるドットの色をクリックします。
3. カラーパレットまたはカラーピッカーから色を選ぶか、16 進数コードを入力します。

{{< img src="/images/runs/run-color-palette.png" alt="project の Workspace でデフォルトの run の色を編集">}}

## run の色をランダム化

テーブル内のすべての run の色をランダム化するには:

1. project サイドバーから **Runs** タブをクリックします。
2. **Name** 列ヘッダーにカーソルを合わせ、三点リーダー（**...**）をクリックし、ドロップダウンメニューから **Randomize run colors** を選択します。

{{% alert %}}
run の色をランダム化するオプションは、並べ替え、フィルタ、検索、グループ化などで run のテーブルを何らかの形で変更した後にのみ利用できます。
{{% /alert %}}


## run の色をリセット




テーブル内のすべての run のデフォルトの色を復元するには:

1. project サイドバーから **Runs** タブをクリックします。
2. **Name** 列ヘッダーにカーソルを合わせ、三点リーダー（**...**）をクリックし、ドロップダウンメニューから **Reset colors** を選択します。

{{< img src="/images/runs/reset-run-colors.png" alt="project の Workspace で run の色をリセット">}}