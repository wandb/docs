---
title: Will wandb slow down my training?
menu:
  support:
    identifier: ja-support-kb-articles-slow_training
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、通常の使用条件下では、トレーニング のパフォーマンスに与える影響は最小限です。通常の使用には、1 秒あたり 1 回未満の頻度で ログ を記録し、データ を 1 ステップあたり数メガバイトに制限することが含まれます。W&B は、ノンブロッキング関数呼び出しで別の プロセス で動作するため、 краткое отсутствие сети または断続的なディスクの読み取り/書き込みの問題がパフォーマンスを中断することはありません。大量のデータ を過度に ログ 記録すると、ディスク I/O の問題が発生する可能性があります。詳細については、サポートにお問い合わせください。
