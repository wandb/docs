---
title: Admin FAQ
displayed_sidebar: default
---

### 팀과 조직의 차이점은 무엇인가요?

팀은 동일한 Projects를 진행하는 사용자 그룹을 위한 협업 workspace이며, 조직은 여러 팀으로 구성될 수 있는 상위 Entity로, 청구 및 계정 관리를 주로 다룹니다.

### 팀과 Entity의 차이점은 무엇인가요? 사용자로서 Entity는 저에게 무엇을 의미하나요?

팀은 동일한 Projects를 진행하는 사용자 그룹을 위한 협업 workspace이며, Entity는 사용자 이름 또는 팀 이름을 지칭합니다. W&B에서 Runs을 기록할 때, Entity를 개인 계정이나 팀 계정으로 설정할 수 있습니다. `wandb.init(entity="example-team")`.

### 팀이란 무엇이며, 이에 대한 추가 정보를 어디에서 찾을 수 있나요?

Teams에 대해 더 알고 싶다면 [teams section](../app/features/teams.md)을 방문하세요.

### 개인 Entity에 비해 팀 Entity에 언제 로그를 남겨야 하나요?

Personal Entities는 2024년 5월 21일 이후 생성된 계정에서는 더 이상 사용할 수 없습니다. W&B는 모든 사용자가 가입 날짜와 관계없이 새로운 Projects를 팀에 로그하여 다른 사람과 결과를 공유할 수 있는 옵션을 갖도록 권장합니다.

### 누가 팀을 생성할 수 있나요? 누가 팀에 사람을 추가하거나 삭제할 수 있나요? 누가 Projects를 삭제할 수 있나요?

다양한 역할과 권한에 관한 정보는 [여기](../app/features/teams.md#team-roles-and-permissions)에서 확인할 수 있습니다.

### 어떤 유형의 역할이 있으며, 그 차이점은 무엇인가요?

다양한 역할과 권한에 대한 정보는 [이 페이지](../app/features/teams.md#team-roles-and-permissions)에서 볼 수 있습니다.

### 서비스 계정이란 무엇이며, 어떻게 우리 팀에 추가하나요?

서비스 계정에 대해 더 알고 싶다면 [이 페이지](./general.md#what-is-a-service-account-and-why-is-it-useful)를 확인하세요.

### 내 조직의 저장된 바이트, 추적된 바이트 및 추적 시간은 어떻게 확인하나요?

* 조직의 저장된 바이트는 `https://<host-url>/usage/<team-name>`에서 확인할 수 있습니다.
* 조직의 추적된 바이트는 `https://<host-url>/usage/<team-name>/tracked`에서 확인할 수 있습니다.
* 조직의 추적 시간은 `https://<host-url>/usage/<team-name>/computehour`에서 확인할 수 있습니다.

### 숨겨진 정말 좋은 기능들은 무엇이며, 어디에서 찾을 수 있나요?

몇몇 기능들이 "Beta Features" 섹션의 feature flag 아래에 숨겨져 있습니다. 사용자 설정 페이지에서 이 기능을 활성화할 수 있습니다.

![Available beta features hidden under a feature flag](/images/technical_faq/beta_features.png)

### 코드가 충돌할 때 어떤 파일을 확인해야 하나요?

영향을 받은 run에 대해서는 `debug.log`과 `debug-internal.log`를 확인해야 합니다. 이 파일들은 코드를 실행하는 동일한 디렉토리의 로컬 폴더 `wandb/run-<date>_<time>-<run-id>/logs` 아래에 있습니다.

### 로컬 인스턴스에서 문제가 있을 때 어떤 파일을 확인해야 하나요?

`Debug Bundle`을 확인해야 합니다. 인스턴스의 관리자는 `/system-admin` 페이지 -> 오른쪽 상단 W&B 아이콘 -> `Debug Bundle`에서 받을 수 있습니다.

![Access System settings page as an Admin of a local instance](/images/technical_faq/local_system_settings.png)
![Download the Debug Bundle as an Admin of a local instance](/images/technical_faq/debug_bundle.png)

### 내가 내 로컬 인스턴스의 관리자라면 어떻게 관리해야 하나요?

인스턴스의 관리자라면, [User Management](../hosting/iam/manage-users.md) 섹션을 참고하여 인스턴스에 사용자를 추가하고 팀을 생성하는 방법을 배우세요.