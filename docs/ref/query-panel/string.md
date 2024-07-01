# string

## チェーン可能な操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[文字列](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 2番目の[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)にサフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | チェックするサブストリング |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わっているかチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わっているかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を見つけます

| 引数 |  |
| :--- | :--- |
| `str` | サブストリングの出現箇所を見つける[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つけるサブストリング |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのインデックスのリスト

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が英数字かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が数値かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が数値かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
先頭の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
小文字に変換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を[文字列](https://docs.wandb.ai/ref/weave/string)のリストにパーティションします

| 引数 |  |
| :--- | :--- |
| `str` | 分割される[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 戻り値
セパレータの前の[文字列](https://docs.wandb.ai/ref/weave/string)、セパレータ、セパレータの後の[文字列](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)にプレフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | プレフィックスを追加する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 追加するプレフィックス |

#### 戻り値
プレフィックスが追加された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
末尾の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換される[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置換するサブストリング |
| `newSub` | 古いサブストリングと置換する新しいサブストリング |

#### 戻り値
置換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

開始と終了のインデックスに基づいて[文字列](https://docs.wandb.ai/ref/weave/string)からサブストリングを抽出します

| 引数 |  |
| :--- | :--- |
| `str` | 抽出する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `begin` | サブストリングの開始インデックス |
| `end` | サブストリングの終了インデックス |

#### 戻り値
サブストリング

<h3 id="string-split"><code>string-split</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を[文字列](https://docs.wandb.ai/ref/weave/string)のリストに分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)両端の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
両端の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | [文字列](https://docs.wandb.ai/ref/weave/string)を大文字に変換する |

#### 戻り値
大文字に変換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの[文字列](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離を計算します

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `str2` | 2番目の[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
2つの[文字列](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離

## リスト操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値が等しくないかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[文字列](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 2番目の[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値が等しいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)にサフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | チェックするサブストリング |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わっているかチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わっているかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を見つけます

| 引数 |  |
| :--- | :--- |
| `str` | サブストリングの出現箇所を見つける[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つけるサブストリング |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのインデックスのリスト

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が英数字かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が英数字かどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がアルファベットかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)が数値かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)が数値かどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
先頭の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を小文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
小文字に変換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を[文字列](https://docs.wandb.ai/ref/weave/string)のリストにパーティションします

| 引数 |  |
| :--- | :--- |
| `str` | 分割される[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 戻り値
セパレータの前の[文字列](https://docs.wandb.ai/ref/weave/string)、セパレータ、セパレータの後の[文字列](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-prepend"><code>string-prepend</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)にプレフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | プレフィックスを追加する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 追加するプレフィックス |

#### 戻り値
プレフィックスが追加された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
末尾の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 置換される[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置換するサブストリング |
| `newSub` | 古いサブストリングと置換する新しいサブストリング |

#### 戻り値
置換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

開始と終了のインデックスに基づいて[文字列](https://docs.wandb.ai/ref/weave/string)からサブストリングを抽出します

| 引数 |  |
| :--- | :--- |
| `str` | 抽出する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `begin` | サブストリングの開始インデックス |
| `end` | サブストリングの終了インデックス |

#### 戻り値
サブストリング

<h3 id="string-split"><code>string-split</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を[文字列](https://docs.wandb.ai/ref/weave/string)のリストに分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[文字列](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレータ |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)のリスト

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェックする[文字列](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[文字列](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)両端の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 削除する[文字列](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
両端の空白が削除された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-upper"><code>string-upper</code></h3>

[文字列](https://docs.wandb.ai/ref/weave/string)を大文字に変換します

| 引数 |  |
| :--- | :--- |
| `str` | [文字列](https://docs.wandb.ai/ref/weave/string)を大文字に変換する |

#### 戻り値
大文字に変換された[文字列](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの[文字列](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離を計算します

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の[文字列](https://docs.wandb.ai/ref/weave/string) |
| `str2` | 2番目の[文字列](https://docs.wandb.ai/ref/weave/string) |

