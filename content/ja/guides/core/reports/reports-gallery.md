---
title: レポートの例
description: レポートギャラリー
menu:
  default:
    identifier: reports-gallery
    parent: reports
weight: 70
---

## メモ: 素早くまとめの可視化を追加する

プロジェクトの進捗中に重要な気づき、今後のアイデア、または達成したマイルストーンをメモしておきましょう。レポート内のすべての experiment run には、パラメータ、メトリクス、ログ、コードへのリンクがついているので、作業の全体像を保存できます。

テキストでメモを書いて、インサイトを示す関連グラフを取り込みましょう。

トレーニング時間の比較を共有する例は、[What To Do When Inception-ResNet-V2 Is Too Slow](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA) の W&B Report をご覧ください。

{{< img src="/images/reports/notes_add_quick_summary.png" alt="Quick summary notes" max-width="90%">}}

複雑なコードベースからベストな事例を簡単に参照・再利用できるように保存できます。Lyft データセットから LIDAR 点群を可視化し、3D バウンディングボックスで注釈を付ける例は、[LIDAR point clouds](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg) の W&B Report をご覧ください。

{{< img src="/images/reports/notes_add_quick_summary_save_best_examples.png" alt="Save best examples" max-width="90%" >}}

## コラボレーション: 学びを同僚と共有する

プロジェクトの開始方法を伝えたり、これまでの観察結果を共有したり、最新の学びをまとめてみましょう。同僚はパネルやレポート末尾のコメント機能を使って提案やディスカッションができます。

動的な設定を含めることで、同僚自身が探索し、さらなる洞察を得て、次のステップをより良く計画できます。以下の例では、3種類の experiment が個別に可視化、比較、平均化できます。

ベンチマークの初回 run と観察を共有する方法の例は、[SafeLife benchmark experiments](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) の W&B Report をご参照ください。

{{< img src="/images/reports/intro_collaborate1.png" alt="SafeLife benchmark report" >}}

{{< img src="/images/reports/intro_collaborate2.png" alt="Experiment comparison view" >}}

スライダーや設定可能なメディアパネルを使って、モデルの結果やトレーニングの進捗を魅力的に紹介してみましょう。[Cute Animals and Post-Modern Style Transfer: StarGAN v2 for Multi-Domain Image Synthesis](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ) では、スライダー付きの W&B Report の例をご覧いただけます。

{{< img src="/images/reports/intro_collaborate3.png" alt="StarGAN report with sliders" >}}

{{< img src="/images/reports/intro_collaborate4.png" alt="Interactive media panels" >}}

## 作業ログ: 試した内容や次のステップを記録する

experiment での考察や学び、注意点、次にやることを随時記録し、すべてを一箇所に整理しましょう。スクリプト以外の大切な部分も「ドキュメント化」できます。[Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc) の W&B Report では、学びをどうレポートできるかの例をご覧いただけます。

{{< img src="/images/reports/intro_work_log_1.png" alt="Text disambiguation report" >}}

プロジェクトのストーリーを記録することで、後から自分や他の人が、どのように・なぜそのモデルが開発されたかを理解できるようになります。[The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ) の W&B Report をご参照ください。

{{< img src="/images/reports/intro_work_log_2.png" alt="Driver's seat project report" >}}

[Learning Dexterity End-to-End Using W&B Reports](https://bit.ly/wandb-learning-dexterity) では、OpenAI Robotics チームが W&B Reports を活用して大規模な機械学習プロジェクトをどう進めたのかを探索した事例を紹介しています。