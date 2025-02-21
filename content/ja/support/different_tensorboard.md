---
title: How is W&B different from TensorBoard?
menu:
  support:
    identifier: ja-support-different_tensorboard
tags:
- tensorboard
toc_hide: true
type: docs
---

W&B は TensorBoard と連携し、 実験管理 ツールを改善します。創業者たちは、TensorBoard の ユーザー が直面する一般的な不満に対処するために W&B を作成しました。主な改善点は次のとおりです。

1.  **Model Reproducibility**: W&B は、 実験 、探索、および モデル の再現を容易にします。メトリクス、 ハイパーパラメーター 、 コード の バージョン をキャプチャし、 モデル の チェックポイント を保存して、 再現性 を確保します。

2.  **Automatic Organization**: W&B は、試行されたすべての モデル の概要を提供することにより、 プロジェクト の引き継ぎと休暇を効率化します。古い 実験 を再実行するのを防ぐことで、時間を節約します。

3.  **Quick Integration**: W&B を 5 分で プロジェクト に統合します。無料のオープンソース Python パッケージ をインストールし、数行の コード を追加します。ログ に記録された メトリクス とレコードは、各 モデル の run と共に表示されます。

4.  **Centralized Dashboard**: トレーニング がローカル、ラボ クラスター 、または クラウド スポット インスタンス のどこで行われても、一貫した ダッシュボード に アクセス できます。異なるマシン間で TensorBoard ファイル を管理する必要がなくなります。

5.  **Robust Filtering Table**: さまざまな モデル からの 結果 を効率的に検索、フィルタリング、ソート、およびグループ化します。さまざまなタスクに最適な モデル を簡単に特定できます。これは、TensorBoard が大規模な プロジェクト で苦労することが多い分野です。

6.  **Collaboration Tools**: W&B は、複雑な 機械学習 プロジェクト の コラボレーション を強化します。プロジェクト のリンクを共有し、プライベート Teams を利用して 結果 を共有します。インタラクティブな 可視化 と ワークログ またはプレゼンテーション用の markdown による説明を含む Reports を作成します。
