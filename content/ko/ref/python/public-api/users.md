---
title: 사용자
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-users
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/users.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API를 사용하여 사용자와 API 키를 관리할 수 있습니다.

이 모듈은 W&B 사용자 및 API 키 관리를 위한 클래스를 제공합니다.



**참고:**

> 이 모듈은 W&B Public API의 일부로, 사용자와 인증 관리를 위한 메소드를 제공합니다. 일부 작업은 관리자 권한이 필요할 수 있습니다.



---

## <kbd>class</kbd> `User`
W&B 사용자에 대한 인증 및 관리 기능을 가진 클래스입니다.

이 클래스는 W&B 사용자의 생성, API 키 관리, 팀 멤버십 엑세스 등 사용자 관리 메소드를 제공합니다. 사용자의 속성 관리를 위해 Attrs를 상속받습니다.



**ARG:**
 
 - `client`:  (`wandb.apis.internal.Api`) 사용할 클라이언트 인스턴스 
 - `attrs`:  (dict) 사용자 속성



**참고:**

> 일부 작업은 관리자 권한이 필요합니다.

### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```






---

### <kbd>property</kbd> User.api_keys

사용자와 연결된 API 키 이름들의 목록입니다.



**반환값:**
 
 - `list[str]`:  사용자와 연결된 API 키 이름들의 리스트입니다. 사용자가 API 키가 없거나 API 키 데이터가 아직 로딩되지 않았다면 빈 리스트를 반환합니다.

---

### <kbd>property</kbd> User.teams

해당 사용자가 멤버로 속한 팀 이름들의 목록입니다.



**반환값:**
 
 - `list` (list):  사용자가 소속된 팀의 이름 리스트. 사용자가 팀 멤버십이 없거나 팀 데이터가 로딩되지 않았다면 빈 리스트를 반환합니다.

---

### <kbd>property</kbd> User.user_api

사용자의 인증 정보를 사용하여 생성된 api 인스턴스입니다.



---

### <kbd>classmethod</kbd> `User.create`

```python
create(api, email, admin=False)
```

새로운 사용자를 생성합니다.



**ARG:**
 
 - `api` (`Api`):  사용할 api 인스턴스 
 - `email` (str):  팀의 이름 
 - `admin` (bool):  해당 사용자가 전체 인스턴스 관리자인지 여부



**반환값:**
 `User` 오브젝트

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

사용자의 API 키를 삭제합니다.



**ARG:**
 
 - `api_key` (str):  삭제할 API 키의 이름입니다. 이 값은 `api_keys` 속성에서 반환되는 이름 중 하나여야 합니다.



**반환값:**
 성공 여부를 나타내는 Boolean



**예외 발생:**
 api_key를 찾지 못한 경우 ValueError 발생

---

### <kbd>method</kbd> `User.generate_api_key`

```python
generate_api_key(description=None)
```

새로운 API 키를 생성합니다.



**ARG:**
 
 - `description` (str, optional):  새 API 키에 대한 설명입니다. 이 필드는 API 키의 사용 목적을 구분하는 데 사용할 수 있습니다.



**반환값:**
 새로 생성된 API 키, 실패 시 None 반환
