---
title: Use Prometheus monitoring
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

W&B Server와 함께 [Prometheus](https://prometheus.io/docs/introduction/overview/)를 사용하세요. Prometheus 설치는 [kubernetes ClusterIP service](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)로 노출됩니다.

{{% alert color="secondary" %}}
Prometheus 모니터링은 [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})에서만 사용할 수 있습니다.
{{% /alert %}}

Prometheus 메트릭 엔드포인트 (`/metrics`)에 엑세스하려면 아래 절차를 따르세요.

1. Kubernetes CLI 툴킷인 [kubectl](https://kubernetes.io/docs/reference/kubectl/)을 사용하여 클러스터에 연결합니다. 자세한 내용은 Kubernetes의 [클러스터 엑세스](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) 문서를 참조하세요.
2. 다음 코맨드를 사용하여 클러스터의 내부 어드레스를 찾습니다:

    ```bash
    kubectl describe svc prometheus
    ```

3. [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)를 사용하여 Kubernetes 클러스터에서 실행 중인 컨테이너 내부에서 셸 세션을 시작합니다. `<internal address>/metrics`에서 엔드포인트를 연결합니다.

   아래 코맨드를 복사하여 터미널에서 실행하고 `<internal address>`를 내부 어드레스로 바꿉니다:

   ```bash
   kubectl exec <internal address>/metrics
   ```

테스트 Pod가 시작되면 네트워크의 모든 것에 엑세스하기 위해 exec 할 수 있습니다:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

여기에서 네트워크 내부로 엑세스를 유지하거나 Kubernetes 노드포트 서비스로 직접 노출하도록 선택할 수 있습니다.
