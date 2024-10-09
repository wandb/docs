---
title: Artifact data privacy and compliance
description: W&B 파일이 기본적으로 저장되는 위치를 알아보세요. 민감한 정보를 저장하고 보관하는 방법을 탐색하세요.
displayed_sidebar: default
---

파일은 아티팩트를 로그할 때 W&B에서 관리하는 Google Cloud 버킷에 업로드됩니다. 버킷의 내용은 저장 중에도 전송 중에도 암호화됩니다. 아티팩트 파일은 해당 프로젝트에 엑세스 권한이 있는 사용자에게만 표시됩니다.

![GCS W&B 클라이언트 서버 다이어그램](/images/artifacts/data_and_privacy_compliance_1.png)

아티팩트의 버전을 삭제하면, 이는 우리의 데이터베이스에 소프트 삭제로 표시되며 저장 비용에서 제거됩니다. 전체 아티팩트를 삭제하면 영구 삭제를 위해 대기열에 추가되며 W&B 버킷에서 모든 내용이 제거됩니다. 파일 삭제에 관한 특정 요구 사항이 있는 경우 [Customer Support](mailto:support@wandb.com)에 문의하세요.

다중 테넌트 환경에 있을 수 없는 민감한 데이터셋의 경우, 클라우드 버킷에 연결된 프라이빗 W&B 서버나 _reference artifacts_를 사용할 수 있습니다. Reference artifacts는 파일 내용을 W&B로 전송하지 않고 프라이빗 버킷에 대한 참조를 추적합니다. Reference artifacts는 버킷이나 서버에 있는 파일에 대한 링크를 유지합니다. 다시 말해, W&B는 파일 자체가 아니라 파일과 관련된 메타데이터만 추적합니다.

![W&B 클라이언트 서버 클라우드 다이어그램](/images/artifacts/data_and_privacy_compliance_2.png)

reference artifact를 생성하는 방법은 비reference artifact를 생성하는 것과 유사합니다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

대안에 대해 논의하려면 [contact@wandb.com](mailto:contact@wandb.com)으로 연락하여 프라이빗 클라우드 및 온프레미스 설치에 대해 상담하세요.