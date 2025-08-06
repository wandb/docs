---
title: レポート
description: 機械学習プロジェクトのためのプロジェクト管理およびコラボレーションツール
menu:
  default:
    identifier: reports
    parent: core
weight: 3
url: guides/reports
cascade:
- url: guides/reports/:filename
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb" >}}

W&B Reports を使うことで以下が可能です：
- Runs の整理
- 可視化の埋め込みと自動化
- 学びの記述
- コラボレーターとアップデートを共有（LaTeX の zip ファイルや PDF でも共有可能）

次の画像は、トレーニング中に W&B に記録されたメトリクスから作成されたレポートの一部です。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="W&B report with benchmark results" max-width="90%" >}}

上記画像が使われているレポートは[こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)からご覧いただけます。

## 仕組み
数回のクリックで共同編集可能なレポートを作成できます。

1. W&B アプリ内の該当 Project Workspace にアクセスします。
2. ワークスペース右上の **Create report** ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="Create report button" max-width="90%">}}

3. **Create Report** というタイトルのモーダルが表示されます。レポートに追加したいチャートやパネルを選択します（後で追加・削除も可能です）。
4. **Create report** をクリックします。
5. お好みの状態になるまでレポートを編集します。
6. **Publish to project** をクリックします。
7. **Share** ボタンからコラボレーターとレポートを共有できます。

レポートのインタラクティブな作成方法や W&B Python SDK を使ったプログラムによる作成方法は、[Create a report]({{< relref "./create-a-report.md" >}}) のページをご覧ください。

## 開始方法
用途に合わせて、W&B Reports の利用を始めるためのリソースをご活用ください：

* W&B Reports の概要は [動画デモ](https://www.youtube.com/watch?v=2xeJIv_K_eI) をご覧ください。
* ライブレポートの例は [Reports gallery]({{< relref "./reports-gallery.md" >}}) をご覧ください。
* [Programmatic Workspaces]({{< relref "/tutorials/workspaces.md" >}}) チュートリアルで Workspace の作成方法やカスタマイズ方法を学べます。
* 厳選されたレポートは [W&B Fully Connected](https://wandb.me/fc) でご覧ください。

## ベストプラクティス・TIPS

Experiments やログに関するベストプラクティスやヒントについては、[Best Practices: Reports](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#reports) をご確認ください。