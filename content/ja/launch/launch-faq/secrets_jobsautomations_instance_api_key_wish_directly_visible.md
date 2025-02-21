---
title: Can you specify secrets for jobs/automations? For instance, an API key which
  you do not wish to be directly visible to users?
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。以下の手順に従ってください。

1. 次の コマンド を使用して、run 用の Kubernetes シークレットを指定された名前空間に作成します:
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成したら、run の開始時にシークレットを挿入するようにキューを設定します。クラスタ 管理者のみがシークレットを表示でき、エンド ユーザー は表示できません。
