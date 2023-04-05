---
slug: /guides/reports
description: Project management and collaboration tools for machine learning projects
---

# レポートを使用した共同作業

W&Bレポートを使ってRunを体系化し、可視化の埋め込みと自動化を行い、発見事項を説明し、最新情報を共同作業者と共有します。


:::info
レポートの[デモ動画](https://www.youtube.com/watch?v=2xeJIv\_K\_eI)を鑑賞し、[W&B Fully Connected](http://wandb.me/fc)のキュレートされたレポートをお読みください。
:::

<!-- {% embed url="https://www.youtube.com/watch?v=2xeJIv_K_eI" %} -->

## レポートの一般的なユースケース​

1. **コラボレーション**: 発見事項を同僚と共有しましょう。
2. **作業ログ**: 試したことを追跡し、次のステップを計画します。
3. **自動化された可視化**: レポートAPIを使って、モデル分析をモデルCI/CD開発フローに統合します。

### 注：簡単な要約と共に可視化を追加

重要な発見事項、将来の作業用のアイデア、またはプロジェクトの開発中に到達したマイルストーンをキャプチャします。レポート内の実験runはすべて、パラメーター、メトリクス、ログ、およびコードにリンクされるため、作業の完全なコンテキストを保存できます。


![](/images/reports/notes_add_quick_summary.png)

複雑なコードベースから最善の例を保存し、その後の作業時に簡単に参照できます（例: [LIDAR point clouds](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg))

![](/images/reports/notes_add_quick_summary_save_best_examples.png)

### コラボレーション：発見事項を同僚と共有​

プロジェクトに取りかかり、これまでの観察内容を共有し、最新の発見事項を合成する方法を説明します。同僚は提案をしたり、パネル上やレポートの最後の部分でコメントを使って詳細について話し合ったりすることができます。

動的な設定を含めることで、同僚は自分で探索したり、追加のインサイトを入手したり、次のステップを適切に計画したりすることができます。この例では、3種類の実験を個別に可視化、比較し、平均することができます。

![](/images/reports/intro_collaborate1.png)

![](/images/reports/intro_collaborate2.png)

スライダーと設定可能なメディアパネルを使って、モデルの結果やトレーニングの進捗状況を示します（例: [Cute Animals and Post-Modern Style Transfer: StarGAN v2 for Multi-Domain Image Synthesis](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ))。

![](/images/reports/intro_collaborate3.png)

![](/images/reports/intro_collaborate4.png)

### 作業ログ：試したことを追跡し、次のステップを計画

プロジェクトに取り組みながら、実験に関する考察、発見事項、理解した内容および次のステップを書き留め、すべての内容を1か所で整理します。これによって、スクリプトの外で、重要事項をすべて`文書化`することができます（例: [Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc))。

![](/images/reports/intro_work_log_1.png)

あなたと他の人が後で参照してモデルの開発方法と開発理由を理解できるように、プロジェクトのストーリーを伝えましょう （例: [The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ))。

![](/images/reports/intro_work_log_2.png)

See the [Learning Dexterity End-to-End Using Weights & Biases Reports](https://bit.ly/wandb-learning-dexterity) for an example of how W&B Reports were used to explore how the OpenAI Robotics team used Weights & Biases Reports to run massive machine learning projects.

<!-- Once you have [experiments in W&B](../../quickstart.md), easily visualize results in reports. Here's a quick overview video. -->

<!-- {% embed url="https://www.youtube.com/watch?v=o2dOSIDDr1w" %} -->
