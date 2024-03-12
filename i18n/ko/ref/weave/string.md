
# 문자열

## 연결 가능한 연산
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

두 값의 불일치를 결정합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 같지 않은지 여부입니다.

<h3 id="string-add"><code>string-add</code></h3>

두 [문자열](https://docs.wandb.ai/ref/weave/string)을 연결합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 두 번째 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
연결된 [문자열](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

두 값의 동등성을 결정합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 동일한지 여부입니다.

<h3 id="string-append"><code>string-append</code></h3>

접미사를 [문자열](https://docs.wandb.ai/ref/weave/string)에 추가합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접미사를 추가할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 추가할 접미사 |

#### 반환 값
접미사가 추가된 [문자열](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 특정 부분 문자열을 포함하는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 부분 문자열을 포함하는지 여부입니다.

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 특정 접미사로 끝나는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 찾을 접미사 |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 접미사로 끝나는지 여부입니다.

<h3 id="string-findAll"><code>string-findAll</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)에서 부분 문자열의 모든 발생을 찾습니다.

| 인수 |  |
| :--- | :--- |
| `str` | 부분 문자열의 발생을 찾을 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 찾을 부분 문자열 |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)에서 부분 문자열의 인덱스 목록입니다.

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 영숫자인지 여부입니다.

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 알파벳인지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 알파벳인지 여부입니다.

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 숫자인지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 숫자인지 여부입니다.

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

앞쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 공백을 제거할 [문자열](https://docs.wandb.ai/ref/weave/string)입니다. |

#### 반환 값
공백이 제거된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-len"><code>string-len</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)의 길이를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)의 길이입니다.

<h3 id="string-lower"><code>string-lower</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)을 소문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 소문자로 변환할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
소문자로 변환된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-partition"><code>string-partition</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)을 구분자를 기준으로 분할하여 [문자열](https://docs.wandb.ai/ref/weave/string)의 _목록_으로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 분할 기준이 될 구분자 |

#### 반환 값
구분자 이전의 [문자열](https://docs.wandb.ai/ref/weave/string), 구분자, 구분자 이후의 [문자열](https://docs.wandb.ai/ref/weave/string)로 구성된 _목록_입니다.

<h3 id="string-prepend"><code>string-prepend</code></h3>

접두사를 [문자열](https://docs.wandb.ai/ref/weave/string)에 추가합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 접두사를 추가할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 추가할 접두사 |

#### 반환 값
접두사가 추가된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

뒤쪽 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 공백을 제거할 [문자열](https://docs.wandb.ai/ref/weave/string)입니다. |

#### 반환 값
공백이 제거된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-replace"><code>string-replace</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)에서 부분 문자열의 모든 발생을 다른 부분 문자열로 교체합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 내용을 교체할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 교체할 부분 문자열 |
| `newSub` | 옛 부분 문자열을 교체할 새 부분 문자열 |

#### 반환 값
교체된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-slice"><code>string-slice</code></h3>

시작 인덱스와 끝 인덱스를 기준으로 [문자열](https://docs.wandb.ai/ref/weave/string)을 부분 문자열로 자릅니다.

| 인수 |  |
| :--- | :--- |
| `str` | 자를 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 부분 문자열의 시작 인덱스 |
| `end` | 부분 문자열의 끝 인덱스 |

#### 반환 값
부분 문자열입니다.

<h3 id="string-split"><code>string-split</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)을 구분자를 기준으로 [문자열](https://docs.wandb.ai/ref/weave/string)의 _목록_으로 분할합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 분할할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 분할 기준이 될 구분자 |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)의 _목록_입니다.

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)이 특정 접두사로 시작하는지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 확인할 [문자열](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 찾을 접두사 |

#### 반환 값
[문자열](https://docs.wandb.ai/ref/weave/string)이 접두사로 시작하는지 여부입니다.

<h3 id="string-strip"><code>string-strip</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)의 양쪽 끝에서 공백을 제거합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 공백을 제거할 [문자열](https://docs.wandb.ai/ref/weave/string)입니다. |

#### 반환 값
공백이 제거된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-upper"><code>string-upper</code></h3>

[문자열](https://docs.wandb.ai/ref/weave/string)을 대문자로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `str` | 대문자로 변환할 [문자열](https://docs.wandb.ai/ref/weave/string) |

#### 반환 값
대문자로 변환된 [문자열](https://docs.wandb.ai/ref/weave/string)입니다.

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

두 [문자열](https://docs.wandb.ai/ref/weave/string) 사이의 레벤슈타인 거리를 계산합니다.

| 인수 |  |
| :--- | :--- |
| `str1` | 첫 번째 [문자열](https://docs.wandb.ai/ref/weave/string). |
| `str2` | 두 번째 [문자열](https://docs.wandb.ai/ref/weave/string). |

#### 반환 값
두 [문자열](https://docs.wandb.ai/ref/weave/string) 사이의 레벤슈타인 거리입니다.