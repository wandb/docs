---
title: wandb はトレーニングを遅くしますか？
menu:
  support:
    identifier: ja-support-kb-articles-slow_training
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
W&B は通常の使用条件下でトレーニングパフォーマンスに最小限の影響しか与えません。通常の使用には、1 秒に 1 回以下の頻度でログを作成し、1 ステップあたり数メガバイト以内にデータを制限することが含まれます。W&B は非ブロッキング関数呼び出しを使用して別のプロセスで動作し、短時間のネットワーク障害やディスクの読み書きの断続的な問題がパフォーマンスを妨げないようにします。大量のデータを過剰にログすると、ディスク I/O の問題につながる可能性があります。詳細については、サポートにお問い合わせください。