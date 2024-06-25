
# string

## チェイン可能な操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が不等であるかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初に比較する値。 |
| `rhs` | 次に比較する値。 |

#### 戻り値
2つの値が不等であるかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 次の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初に比較する値。 |
| `rhs` | 次に比較する値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)に接尾辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加する接尾辞 |

#### 戻り値
接尾辞が追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含むかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 確認する部分文字列 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含む場合

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が特定の接尾辞で終わるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 確認する接尾辞 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がその接尾辞で終わる場合

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のすべての出現箇所を検索します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現箇所を検索する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 検索する部分文字列 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のインデックスのリスト

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が英数字である場合

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がアルファベットであるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がアルファベットである場合

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が数値であるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が数値である場合

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

前方の空白文字を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
小文字に変換された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を[string](https://docs.wandb.ai/ref/weave/string)のリストにパーティションします

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割する区切り文字 |

#### 戻り値
区切り文字の前の[string](https://docs.wandb.ai/ref/weave/string)、区切り文字、および区切り文字の後の[string](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)に接頭辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 接頭辞を追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 追加する接頭辞 |

#### 戻り値
接頭辞が追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

後方の空白文字を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列をすべて置き換えます

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置き換える[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置き換える部分文字列 |
| `newSub` | 元の部分文字列と置き換える部分文字列 |

#### 戻り値
置き換えられた[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて[string](https://docs.wandb.ai/ref/weave/string)から部分文字列をスライスします

| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | 部分文字列の開始インデックス |
| `end` | 部分文字列の終了インデックス |

#### 戻り値
部分文字列

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を[string](https://docs.wandb.ai/ref/weave/string)のリストに分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割する区切り文字 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が特定の接頭辞で始まるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 確認する接頭辞 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が接頭辞で始まる場合

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の両端から空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
大文字に変換された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの[string](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離を計算します。

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[string](https://docs.wandb.ai/ref/weave/string) |
| `str2` | 次の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
2つの[string](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離


## リスト操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が不等であるかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初に比較する値。 |
| `rhs` | 次に比較する値。 |

#### 戻り値
2つの値が不等であるかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 次の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初に比較する値。 |
| `rhs` | 次に比較する値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)に接尾辞を追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加する接尾辞 |

#### 戻り値
接尾辞が追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含むかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 確認する部分文字列 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が部分文字列を含む場合

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が特定の接尾辞で終わるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 確認する接尾辞 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がその接尾辞で終わる場合

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のすべての出現箇所を検索します

| 引数 |  |
| :--- | :--- |
| `str` | 部分文字列の出現箇所を検索する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 検索する部分文字列 |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)内の部分文字列のインデックスのリスト

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が英数字である場合

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がアルファベットであるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がアルファベットである場合

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が数値であるかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `str` | 確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が数値である場合

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

前方の空白文字を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len"></code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | 長さを確認する[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)