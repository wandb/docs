---
title: Reports
description: 機械学習プロジェクト向けのプロジェクト管理およびコラボレーションツール
cascade:
- url: guides/reports/:filename
menu:
  default:
    identifier: ja-guides-core-reports-_index
    parent: core
url: guides/reports
weight: 3
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb" >}}

W&B Reports を使うと次のことができます：
- Runs を整理する。
- 可視化を埋め込み、自動化する。
- 自分の findings を記述する。
- LaTeX の zip ファイルや PDF として、コラボレーターにアップデートを共有することができる。

以下の画像は、トレーニング中に W&B に記録された metrics をもとに作成された report の一部です。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="W&B report with benchmark results" max-width="90%" >}}

上記画像が掲載されている report は[こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)からご覧いただけます。

## 仕組み
数クリックで、コラボレーション可能な report を作成できます。

1. W&B App で自分の W&B Project Workspace に移動します。
2. Workspace の右上にある **Create report** ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="Create report button" max-width="90%">}}

3. **Create Report** というタイトルのモーダルが表示されます。report に追加したいチャートやパネルを選択してください。（後からチャートやパネルの追加・削除も可能です。）
4. **Create report** をクリックします。
5. レポートを好みの内容に編集します。
6. **Publish to project** をクリックします。
7. **Share** ボタンを押して、コラボレーターと report を共有しましょう。

W&B Python SDK を使って対話的・プログラム的に report を作成する方法については、[Create a report]({{< relref path="./create-a-report.md" lang="ja" >}}) のページをご覧ください。

## 開始方法
ユースケースに応じて、以下のリソースから W&B Reports の利用を始めてみましょう：

* W&B Reports の概要を知りたい方は、[ビデオデモンストレーション](https://www.youtube.com/watch?v=2xeJIv_K_eI) をご覧ください。
* ライブ Reports の事例は、[Reports ギャラリー]({{< relref path="./reports-gallery.md" lang="ja" >}}) を参照してください。
* Workspace の作成・カスタマイズ方法を学ぶには、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルをお試しください。
* [W&B Fully Connected](https://wandb.me/fc) でキュレーションされた Reports を読むことができます。

## ベストプラクティス・TIPS

Experiments やログ記録のためのベストプラクティスやヒントについては、[Best Practices: Reports](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#reports) をご覧ください。