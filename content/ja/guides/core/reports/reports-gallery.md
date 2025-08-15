---
title: レポートの例
description: レポートギャラリー
menu:
  default:
    identifier: ja-guides-core-reports-reports-gallery
    parent: reports
weight: 70
---

## メモ：要約とともに可視化を追加

プロジェクト開発中に得られた重要な観察結果、今後のアイデア、またはマイルストーンを記録しましょう。あなたのレポート内のすべての Runs は、各パラメータ、メトリクス、ログ、コードへのリンクが付いているため、作業内容の全コンテキストを保存できます。

気づきやアイデアを書き留め、関連するチャートを追加してインサイトをわかりやすく可視化できます。

トレーニング時間の比較方法の例については、[What To Do When Inception-ResNet-V2 Is Too Slow](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA) の W&B Report をご覧ください。

{{< img src="/images/reports/notes_add_quick_summary.png" alt="クイック要約のメモ" max-width="90%">}}

複雑なコードベースの中から優れた例を保存し、簡単に振り返ったり再利用できるようにしましょう。Lyft dataset の LIDAR ポイントクラウドを可視化し、3D バウンディングボックスでアノテーションした例は [LIDAR point clouds](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg) の W&B Report で確認できます。

{{< img src="/images/reports/notes_add_quick_summary_save_best_examples.png" alt="優れた例の保存" max-width="90%" >}}

## コラボレーション：学びを同僚と共有

プロジェクトの開始方法を説明したり、これまでの観察内容や最新の学びをまとめて共有しましょう。同僚は、各パネルやレポート末尾のコメント機能を使って提案やディスカッションができます。

動的な設定なども追加すれば、同僚自身が試行錯誤しながら新たな知見を得たり、次のステップを計画しやすくなります。下記の例では、3 種類の Experiments を個別表示・比較・平均化することができます。

ベンチマークの初回 Runs や観察内容の共有例は、[SafeLife benchmark experiments](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) の W&B Report をご覧ください。

{{< img src="/images/reports/intro_collaborate1.png" alt="SafeLife ベンチマークレポート" >}}

{{< img src="/images/reports/intro_collaborate2.png" alt="Experiment の比較ビュー" >}}

スライダーやカスタマイズ可能なメディアパネルを使って、モデルの結果やトレーニングの進行状況を紹介することもできます。スライダーつき W&B Report の例としては、[Cute Animals and Post-Modern Style Transfer: StarGAN v2 for Multi-Domain Image Synthesis](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ) のレポートをご覧ください。

{{< img src="/images/reports/intro_collaborate3.png" alt="スライダー付きの StarGAN レポート" >}}

{{< img src="/images/reports/intro_collaborate4.png" alt="インタラクティブなメディアパネル" >}}

## 作業記録：試行内容の記録と次のステップの計画

Experiments の検討内容、学び、注意点、次のアクションなどを都度記録して、プロジェクトの進行を一箇所で整理しましょう。Script だけでは残せない重要な内容も _“ドキュメント化”_ できます。学びのレポート例としては、[Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc) の W&B Report があります。

{{< img src="/images/reports/intro_work_log_1.png" alt="テキスト曖昧性解消プロジェクトレポート" >}}

プロジェクトのストーリーをまとめておけば、後から自分やチームメンバーが、なぜどのようにモデルが開発されたか理解しやすくなります。[The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ) の W&B Report もご覧ください。

{{< img src="/images/reports/intro_work_log_2.png" alt="ドライバー視点のプロジェクトレポート" >}}

また、[Learning Dexterity End-to-End Using W&B Reports](https://bit.ly/wandb-learning-dexterity) では、OpenAI Robotics チームが W&B Reports を使って大規模な machine learning プロジェクトをどのように推進したかを探索した事例も紹介しています。