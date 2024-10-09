---
title: Configure SSO with LDAP
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B 서버 LDAP 서버로 자격 증명을 인증합니다. 다음 가이드는 W&B Server의 설정을 구성하는 방법을 설명합니다. 필수 및 선택적 설정뿐만 아니라 시스템 설정 UI에서 LDAP 연결을 구성하는 지침을 다룹니다. 또한 어드레스, 기본 식별 이름, 속성과 같은 LDAP 설정의 다양한 입력에 대한 정보를 제공합니다. 이러한 속성은 W&B App UI에서 지정하거나 환경 변수를 사용하여 지정할 수 있습니다. 익명 바인드 또는 관리자 DN 및 비밀번호로 바인드를 설정할 수 있습니다.

:::tip
W&B 관리자 역할만 LDAP 인증을 활성화하고 구성할 수 있습니다.
:::

## LDAP 연결 구성

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'Environment variables', value: 'env'},
    
  ]}>
  <TabItem value="app">

1. W&B App으로 이동합니다.
2. 오른쪽 상단에서 프로필 아이콘을 선택합니다. 드롭다운에서 **System Settings**를 선택합니다.
3. **Configure LDAP Client**를 켭니다.
4. 양식에 세부 정보를 입력합니다. 각 입력에 대한 자세한 내용은 **Configuring Parameters** 섹션을 참조하세요.
5. **Update Settings**를 클릭하여 설정을 테스트합니다. 이것은 W&B 서버와의 테스트 클라이언트/연결을 설정합니다.
6. 연결이 확인되면 **Enable LDAP Authentication**을 켜고 **Update Settings** 버튼을 선택합니다.

  </TabItem>
  <TabItem value="env">

다음 환경 변수를 사용하여 LDAP 연결을 설정합니다:

| 환경 변수                      | 필수      | 예                              |
| ----------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | Yes      | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | Yes      | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | No       | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | No       |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | Yes      | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | No       |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | No       |                                 |
| `LOCAL_LDAP_LOGIN`            | No       |                                 |

각 환경 변수의 정의는 [설정 파라미터](#configuration-parameters) 섹션을 참조하십시오. 명확성을 위해 환경 변수 접두사 `LOCAL_LDAP`는 정의 이름에서 생략되었습니다.

  </TabItem>
</Tabs>

## 설정 파라미터

다음 표는 필수 및 선택적 LDAP 설정을 나열하고 설명합니다.

| 환경 변수          | 정의                                                       | 필수      |
| ------------------ | --------------------------------------------------------- | -------- |
| `ADDRESS`          | W&B Server를 호스팅하는 VPC 내의 LDAP 서버의 어드레스입니다.         | Yes      |
| `BASE_DN`          | 이 디렉토리에 쿼리하기 위해 필요한 루트 경로 검색 시작점입니다.         | Yes      |
| `BIND_DN`          | LDAP 서버에 등록된 관리 사용자의 경로입니다. LDAP 서버가 인증되지 않은 바인딩을 지원하지 않으면 필수입니다. 지정된 경우 W&B Server는 이 사용자로 LDAP 서버에 연결합니다. 그렇지 않으면 W&B Server는 익명 바인딩을 사용하여 연결합니다. | No       |
| `BIND_PW`          | 바인딩을 인증하는 데 사용되는 관리 사용자의 비밀번호입니다. 비워 두면 W&B Server는 익명 바인딩을 사용하여 연결합니다. | No       |
| `ATTRIBUTES`       | 이메일 및 그룹 ID 속성 이름을 콤마로 구분된 문자열 값으로 제공합니다. | Yes      |
| `TLS_ENABLE`       | TLS를 활성화합니다.                                        | No       |
| `GROUP_ALLOW_LIST` | 그룹 허용 목록입니다.                                       | No       |
| `LOGIN`            | W&B Server에 LDAP를 사용하여 인증하도록 지시합니다. `True` 또는 `False`로 설정하십시오. LDAP 설정을 테스트하려면 선택적으로 이 값을 false로 설정하십시오. LDAP 인증을 시작하려면 이 값을 true로 설정하십시오. | No       |