---
title: Can you specify secrets for jobs/automations? For instance, an API key which
  you do not wish to be directly visible to users?
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。以下の手順に従ってください。

1. 次のコマンドを使用して、 run 用の指定された名前空間に Kubernetes シークレットを作成します。
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成したら、 run 開始時にシークレットを注入するようにキューを設定します。クラスター管理者のみがシークレットを表示でき、エンド ユーザー は表示できません。
