---
title: ターゲット環境で Launch は計算リソースを自動でプロビジョニング (そしてスピンダウン) できますか?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_automatically_provision_spin_compute_resources_target_environment
    parent: launch-faq
---

このプロセスは環境に依存します。Amazon SageMaker と Vertex でリソースが提供されます。Kubernetes では、オートスケーラーが需要に基づいてリソースを自動的に調整します。W&B のソリューション アーキテクトが Kubernetes インフラストラクチャーの設定を支援し、再試行、自動スケーリング、およびスポット インスタンス ノード プールの使用を可能にします。サポートについては、support@wandb.com に連絡するか、共有された Slack チャンネルを使用してください。