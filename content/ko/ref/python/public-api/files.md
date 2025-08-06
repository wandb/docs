---
title: 파일
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-files
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API에서 File 오브젝트를 다룹니다.

이 모듈은 W&B에 저장된 파일을 다루기 위한 클래스를 제공합니다.



**예시:**
 ```python
from wandb.apis.public import Api

# 특정 run에서 파일 가져오기
run = Api().run("entity/project/run_id")
files = run.files()

# 파일 다루기
for file in files:
     print(f"File: {file.name}")
     print(f"Size: {file.size} bytes")
     print(f"Type: {file.mimetype}")

     # 파일 다운로드
     if file.size < 1000000:  # 1MB 미만일 때
         file.download(root="./downloads")

     # 대용량 파일의 경우 S3 URI 가져오기
     if file.size >= 1000000:
         print(f"S3 URI: {file.path_uri}")
```



**노트:**

> 이 모듈은 W&B Public API의 일부로, W&B에 저장된 파일을 엑세스하고, 다운로드하고, 관리할 수 있는 메소드들을 제공합니다. 파일은 일반적으로 특정 run과 연결되며, 모델 가중치, 데이터셋, 시각화, 기타 Artifacts를 포함할 수 있습니다.

## <kbd>class</kbd> `Files`
`File` 오브젝트의 iterable 컬렉션입니다.

run 중에 W&B에 업로드된 파일을 엑세스하고 관리할 수 있습니다. 대용량 파일 컬렉션을 반복할 때 자동으로 페이지네이션을 처리합니다.



**예시:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# 예시 run 오브젝트
run = Api().run("entity/project/run-id")

# run의 파일을 반복할 수 있는 Files 오브젝트 생성
files = Files(api.client, run)

# 파일 반복 처리
for file in files:
     print(file.name)
     print(file.url)
     print(file.size)

     # 파일 다운로드
     file.download(root="download_directory", replace=True)
```

### <kbd>method</kbd> `Files.__init__`

```python
__init__(client, run, names=None, per_page=50, upload=False)
```

특정 run의 `File` 오브젝트를 iterable로 만듭니다.



**인자:**
 client: 파일이 담긴 run 오브젝트 run: 파일이 담긴 run 오브젝트 names (list, optional): 파일 이름 목록으로 필터링 per_page (int, optional): 한 페이지에 가져올 파일 개수 upload (bool, optional): `True`일 경우 각 파일의 업로드 URL까지 가져옴


---


### <kbd>property</kbd> Files.length





---