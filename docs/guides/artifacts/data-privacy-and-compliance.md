---
description: Learn where W&B files are stored by default. Explore how to save, store
  sensitive information.
displayed_sidebar: default
---

# 데이터 개인 정보 보호 및 준수

<head>
    <title>아티팩트 데이터 개인 정보 보호 및 준수</title>
</head>

아티팩트를 로그할 때 파일은 W&B가 관리하는 Google 클라우드 버킷에 업로드됩니다. 버킷의 내용은 휴식 중(rest)과 전송 중(transit) 모두 암호화됩니다. 아티팩트 파일은 해당 프로젝트에 액세스할 수 있는 사용자만 볼 수 있습니다.

![GCS W&B 클라이언트 서버 다이어그램](/images/artifacts/data_and_privacy_compliance_1.png)

아티팩트의 버전을 삭제하면 데이터베이스에서 소프트 삭제로 표시되며 저장 비용에서 제거됩니다. 아티팩트 전체를 삭제하면 영구 삭제 대기열에 추가되고 모든 내용이 W&B 버킷에서 제거됩니다. 파일 삭제와 관련하여 특별한 요구사항이 있는 경우 [고객 지원](mailto:support@wandb.com)에 문의하십시오.

다중 테넌트 환경에 민감한 데이터세트가 유지될 수 없는 경우, 클라우드 버킷에 연결된 프라이빗 W&B 서버를 사용하거나 _참조 아티팩트_를 사용할 수 있습니다. 참조 아티팩트는 파일 내용을 W&B로 전송하지 않고 프라이빗 버킷에 대한 참조를 추적합니다. 참조 아티팩트는 버킷이나 서버의 파일에 대한 링크를 유지 관리합니다. 즉, W&B는 파일 자체가 아닌 파일과 관련된 메타데이터만 추적합니다.

![W&B 클라이언트 서버 클라우드 다이어그램](/images/artifacts/data_and_privacy_compliance_2.png)

참조 아티팩트를 비참조 아티팩트와 유사하게 생성하십시오:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

대안에 대해서는 프라이빗 클라우드 및 온-프레미스 설치에 대해 이야기하려면 [contact@wandb.com](mailto:contact@wandb.com)으로 문의하십시오.