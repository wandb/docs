---
title: Reports の例
description: Reports ギャラリー
menu:
  default:
    identifier: ja-guides-core-reports-reports-gallery
    parent: reports
weight: 70
---

## Notes: クイックサマリーで可視化を追加
重要な所見、将来の作業のアイデア、または Project の開発で達成されたマイルストーンを記録します。Report 内のすべての Runs は、それぞれのパラメータ、メトリクス、ログ、コードにリンクされているため、作業の完全なコンテキストを保持できます。
テキストを書き留め、関連するチャートを取り込んで、洞察を説明します。
[What To Do When Inception-ResNet-V2 Is Too Slow](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA) W&B Report を参照して、トレーニング時間の比較を共有する方法の例を確認してください。
{{< img src="/images/reports/notes_add_quick_summary.png" alt="クイックサマリーノート" max-width="90%">}}
複雑なコードベースから最良の例を保存し、簡単に参照して将来のインタラクションに利用できます。[LIDAR point clouds](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg) W&B Report を参照して、Lyft のデータセットからの LIDAR 点群を可視化し、3D バウンディングボックスでアノテーションを付ける方法の例を確認してください。
{{< img src="/images/reports/notes_add_quick_summary_save_best_examples.png" alt="最良の例を保存" max-width="90%" >}}

## Collaboration: 同僚と学びを共有
Project の開始方法を説明し、これまでに観察したことを共有し、最新の学びをまとめます。同僚は、任意のパネル上、または Report の最後にあるコメントを使用して、提案したり詳細を議論したりできます。
動的な設定を含めることで、同僚は自分で探索し、追加の洞察を得て、次のステップをより良く計画できます。この例では、3 種類の Experiments を個別に可視化したり、比較したり、平均を取ったりできます。
[SafeLife benchmark experiments](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) W&B Report を参照して、ベンチマークの最初の Runs と観測結果を共有する方法の例を確認してください。
{{< img src="/images/reports/intro_collaborate1.png" alt="SafeLife ベンチマーク Report" >}}
{{< img src="/images/reports/intro_collaborate2.png" alt="Experiments 比較ビュー" >}}
スライダーと設定可能なメディア パネルを使用して、モデルの結果またはトレーニングの進捗状況を紹介します。スライダー付きの W&B Report の例として、[Cute Animals and Post-Modern Style Transfer: StarGAN v2 for Multi-Domain Image Synthesis](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ) Report を参照してください。
{{< img src="/images/reports/intro_collaborate3.png" alt="スライダー付き StarGAN Report" >}}
{{< img src="/images/reports/intro_collaborate4.png" alt="インタラクティブ メディア パネル" >}}

## Work log: 試したことを追跡し、次のステップを計画
Experiments、学び、注意点、および次のステップについて、Project を進める中で考えを書き留め、すべてを 1 か所に整理します。これにより、スクリプトを超えたすべての重要な要素を「ドキュメント化」できます。[Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc) W&B Report を参照して、学びを報告する方法の例を確認してください。
{{< img src="/images/reports/intro_work_log_1.png" alt="テキスト曖昧性解消 Report" >}}
Project のストーリーを伝え、後で自分や他の人が参照して、モデルがどのように、なぜ開発されたかを理解できるようにします。[The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ) W&B Report を参照して、学びを報告する方法を確認してください。
{{< img src="/images/reports/intro_work_log_2.png" alt="ドライバー視点 Project Report" >}}
[Learning Dexterity End-to-End Using W&B Reports](https://bit.ly/wandb-learning-dexterity) を参照して、OpenAI Robotics チームが W&B Reports を使用して大規模な機械学習プロジェクトを実行した方法を探る例をご覧ください。