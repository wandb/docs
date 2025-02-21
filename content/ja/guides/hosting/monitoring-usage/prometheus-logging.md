---
title: Use Prometheus monitoring
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

W&B サーバー で [Prometheus](https://prometheus.io/docs/introduction/overview/) を使用します。Prometheus のインストールは、[kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225) として公開されます。

{{% alert color="secondary" %}}
Prometheus のモニタリングは、[自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

Prometheus の メトリクス エンドポイント（`/metrics`）にアクセスするには、以下の手順に従ってください。

1. Kubernetes CLI ツールキットである [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使用して、 クラスター に接続します。詳細については、Kubernetes の [クラスター へのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) のドキュメントを参照してください。
2. 次の コマンド で、 クラスター の内部 アドレス を見つけます。

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes クラスター で実行されている コンテナ 内で、[`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使用してシェル セッションを開始します。`<内部 アドレス>/metrics` でエンドポイントにアクセスします。

   次の コマンド をコピーして ターミナル で実行し、`<internal address>` を内部 アドレス に置き換えます。

   ```bash
   kubectl exec <internal address>/metrics
   ```

テスト pod が起動します。これは、ネットワーク内のあらゆるものにアクセスするためだけに exec できます。

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

そこから、ネットワークへの内部 アクセス を維持するか、kubernetes nodeport サービス で自分で公開するかを選択できます。
