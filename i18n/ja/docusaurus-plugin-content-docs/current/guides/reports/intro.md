---
slug: /guides/reports
description: Project management and collaboration tools for machine learning projects
displayed_sidebar: ja
---

# コラボレーティブレポート

W&Bレポートを使用して、Runsを整理し、可視化を埋め込み・自動化し、結果を説明し、共同作業者との最新情報を共有します。


:::info
レポートの[ビデオデモ](https://www.youtube.com/watch?v=2xeJIv\_K\_eI)をチェックするか、[W&B Fully Connected](http://wandb.me/fc)でキュレーションされたレポートを読んでください。
:::

<!-- {% embed url="https://www.youtube.com/watch?v=2xeJIv_K_eI" %} -->

## レポートの典型的なユースケース

1. **コラボレーション**: 同僚と結果を共有します。
2. **ワークログ**: 試したことをトラッキングし、次のステップを計画します。
3. **自動化された可視化**: Report APIを使用して、モデル分析をモデルのCI/CDパイプラインに統合します。

### ノート: クイックサマリーを含む可視化を追加

プロジェクトの開発において重要な観察、将来の作業のアイデア、または達成されたマイルストーンを記録します。レポート内のすべての実験のRunは、パラメータ、メトリクス、ログ、コードへのリンクが含まれるため、作業の全コンテキストを保存できます。

テキストを書き留め、関連するチャートを引き込んで、洞察を示します。

[Inception-ResNet-V2が遅すぎる場合の対処法](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA)のW&Bレポートを参考に、トレーニング時間の比較をどのように共有できるかを確認してください。

![](/images/reports/notes_add_quick_summary.png)
複雑なコードベースから最高の例を保存し、簡単に参照して将来の対話ができるようにします。[LIDARポイントクラウド](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg)のW&Bレポートでは、LyftデータセットからLIDARポイントクラウドを可視化し、3Dバウンディングボックスで注釈を付ける方法を示しています。

![](/images/reports/notes_add_quick_summary_save_best_examples.png)

### コラボレーション：同僚と調査結果を共有する

プロジェクトの開始方法、これまでの観察結果、最新の調査結果を説明します。同僚は、パネルのコメントやレポートの最後にコメントを使用して、提案や詳細を議論できます。

同僚が自分で調査したり、追加の洞察を得たり、次のステップをよりよく計画できるように、動的な設定を含めます。この例では、3つのタイプの実験を独立して可視化し、比較または平均化できます。

[SafeLifeベンチマーク実験](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)のW&Bレポートでは、ベンチマークの最初のランと観察結果を共有する方法を示しています。

![](/images/reports/intro_collaborate1.png)

![](/images/reports/intro_collaborate2.png)

スライダーや設定可能なメディアパネルを使用して、モデルの結果やトレーニングの進捗状況を紹介します。[かわいい動物とポストモダンスタイル変換：マルチドメイン画像合成のためのStarGAN v2](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ)レポートでは、スライダーを使用したW&Bレポートの例を示しています。

![](/images/reports/intro_collaborate3.png)

![](/images/reports/intro_collaborate4.png)

### ワークログ：試したことや次のステップを追跡する

プロジェクトを進めながら、実験や調査結果、注意点や次のステップに関する考えを記録し、すべてを一か所に整理しておきます。これにより、スクリプト以外の重要な要素をすべて「文書化」できます。 [Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc) のW&Bレポートでは、調査結果を報告する方法を示しています。

![](/images/reports/intro_work_log_1.png)

プロジェクトのストーリーを語り、後で自分や他の人がモデルがどのように、なぜ開発されたかを理解するための参照資料として利用できます。[The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ) のW&Bレポートでは、調査結果を報告する方法を示しています。
![](/images/reports/intro_work_log_2.png)

OpenAIロボティクスチームがどのようにしてWeights & Biases Reportsを使って大規模な機械学習プロジェクトを実行しているかを探るために、W&B Reportsがどのように使用されているのかを示す例として[Learning Dexterity End-to-End Using Weights & Biases Reports](https://bit.ly/wandb-learning-dexterity)をご覧ください。

<!-- W&Bで[実験](../../quickstart.md)を行ったら、レポートで簡単に結果を可視化できます。以下にクイックオーバービデオをご紹介します。 -->

<!-- {% embed url="https://www.youtube.com/watch?v=o2dOSIDDr1w" %} -->