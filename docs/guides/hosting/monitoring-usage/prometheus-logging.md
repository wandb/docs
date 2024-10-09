---
title: Use Prometheus monitoring
displayed_sidebar: default
---

[Prometheus](https://prometheus.io/docs/introduction/overview/)를 W&B 서버와 함께 사용하세요. Prometheus 설치는 [kubernetes ClusterIP 서비스](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)로 노출됩니다.

:::important
Prometheus 모니터링은 [자가 관리 인스턴스](../hosting-options/self-managed.md)에서만 사용 가능합니다.
:::

다음 절차에 따라 Prometheus 메트릭 엔드포인트 (`/metrics`)에 엑세스하십시오:

1. Kubernetes CLI 툴킷 [kubectl](https://kubernetes.io/docs/reference/kubectl/)을 사용하여 클러스터에 연결합니다. 보다 자세한 정보는 Kubernetes의 [엑세스 클러스터](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) 문서를 참조하십시오.
2. 클러스터의 내부 어드레스를 찾습니다:

```bash
kubectl describe svc prometheus
```

3. Kubernetes 클러스터 내에서 실행 중인 컨테이너 안에 쉘 세션을 시작하고 [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)으로 엔드포인트 `<internal address>/metrics`에 접근합니다.

   아래 코맨드를 복사하여 터미널에 실행하고 `<internal address>`를 내부 어드레스로 교체하세요:

   ```bash
   kubectl exec <internal address>/metrics
   ```

이전 코맨드는 네트워크 내의 모든 것에 접근할 수 있도록 단순히 접근만 할 수 있는 더미 pod를 시작합니다:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

거기서부터는 네트워크 내부 엑세스를 유지하거나 원하는 경우 Kubernetes 노드포트 서비스를 사용하여 자체적으로 노출할 수 있습니다.