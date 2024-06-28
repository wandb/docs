
# string

## 連鎖可能な操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値の不等性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 2番目の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値の等価性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)へサフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がサブストリングを含むかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | チェックするサブストリング |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を見つけます

| 引数 |  |
| :--- | :--- |
| `str` | サブストリングの出現箇所を見つける[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つけるサブストリング |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)内のサブストリングのインデックスの_リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がアルファベットだけで構成されているかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がアルファベットだけで構成されているかどうか

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が数字かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が数字であるかどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 剥がす対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
先頭の空白が削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-len"><code>string-len</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の長さを返します

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

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

[string](https://docs.wandb.ai/ref/weave/string)を[string](https://docs.wandb.ai/ref/weave/string)の_リスト_にパーティション分割します。

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレーター |

#### 戻り値
セパレーターの前の[string](https://docs.wandb.ai/ref/weave/string)、セパレーター、およびセパレーターの後の[string](https://docs.wandb.ai/ref/weave/string)からなる[string](https://docs.wandb.ai/ref/weave/string)の_リスト_

<h3 id="string-prepend"><code>string-prepend</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)にプレフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | プレフィックスを追加する[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 追加するプレフィックス |

#### 戻り値
プレフィックスが追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 剥がす対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
末尾の空白が削除された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を置換します

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置換する[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置換するサブストリング |
| `newSub` | 古いサブストリングと置換するサブストリング |

#### 戻り値
置換された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて[string](https://docs.wandb.ai/ref/weave/string)をサブストリングにスライスします

| 引数 |  |
| :--- | :--- |
| `str` | スライスする[string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | サブストリングの開始インデックス |
| `end` | サブストリングの終了インデックス |

#### 戻り値
サブストリング

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)を[string](https://docs.wandb.ai/ref/weave/string)の_リスト_に分割します

| 引数 |  |
| :--- | :--- |
| `str` | 分割する[string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレーター |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)の_リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)の両端から空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 剥がす対象の[string](https://docs.wandb.ai/ref/weave/string) |

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
| `str2` | 2番目の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
2つの[string](https://docs.wandb.ai/ref/weave/string)間のレーベンシュタイン距離

## リスト操作
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

2つの値の不等性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="string-add"><code>string-add</code></h3>

2つの[string](https://docs.wandb.ai/ref/weave/string)を連結します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[string](https://docs.wandb.ai/ref/weave/string) |
| `rhs` | 2番目の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
連結された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-equal"><code>string-equal</code></h3>

2つの値の等価性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="string-append"><code>string-append</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)へサフィックスを追加します

| 引数 |  |
| :--- | :--- |
| `str` | 追加先の[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | 追加するサフィックス |

#### 戻り値
サフィックスが追加された[string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-contains"><code>string-contains</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がサブストリングを含むかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | チェックするサブストリング |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がサブストリングを含んでいるかどうか

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |
| `suffix` | チェックするサフィックス |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がサフィックスで終わるかどうか

<h3 id="string-findAll"><code>string-findAll</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)内のサブストリングのすべての出現箇所を見つけます

| 引数 |  |
| :--- | :--- |
| `str` | サブストリングの出現箇所を見つける[string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 見つけるサブストリング |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)内のサブストリングのインデックスの_リスト_

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が英数字であるかどうか

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

[string](https://docs.wandb.ai/ref/weave/string)がアルファベットだけで構成されているかどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)がアルファベットだけで構成されているかどうか

<h3 id="string-isNumeric"><code>string-isNumeric"></code></h3>

[string](https://docs.wandb.ai/ref/weave/string)が数字かどうかをチェックします

| 引数 |  |
| :--- | :--- |
| `str` | チェック対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string)が数字であるかどうか

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

先頭の空白を削除します

| 引数 |  |
| :--- | :--- |
| `str` | 剥がす対象の[string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
ストリップされた [string](https://docs.wandb.ai/ref/weave/string)。

<h3 id="string-len"><code>string-len</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) の長さを返す

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string) の長さ

<h3 id="string-lower"><code>string-lower</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) を小文字に変換する

| 引数 |  |
| :--- | :--- |
| `str` | 小文字に変換する [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
小文字に変換された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-partition"><code>string-partition</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) を _リスト_ にパーティションする

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレーター |

#### 戻り値
_リスト_ の [strings](https://docs.wandb.ai/ref/weave/string): セパレーターの前の [string](https://docs.wandb.ai/ref/weave/string)、セパレーター、およびセパレーターの後の [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-prepend"><code>string-prepend</code></h3>

プレフィックスを [string](https://docs.wandb.ai/ref/weave/string) に付加する

| 引数 |  |
| :--- | :--- |
| `str` | 付加される [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | 付加するプレフィックス |

#### 戻り値
プレフィックスが付加された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

末尾の空白を取り除く

| 引数 |  |
| :--- | :--- |
| `str` | ストリップする [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
ストリップされた [string](https://docs.wandb.ai/ref/weave/string)。

<h3 id="string-replace"><code>string-replace</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) 内のサブストリングをすべて置換する

| 引数 |  |
| :--- | :--- |
| `str` | 内容を置換する [string](https://docs.wandb.ai/ref/weave/string) |
| `sub` | 置換するサブストリング |
| `newSub` | 古いサブストリングと置き換える新しいサブストリング |

#### 戻り値
置換された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-slice"><code>string-slice</code></h3>

開始インデックスと終了インデックスに基づいて、[string](https://docs.wandb.ai/ref/weave/string) をサブストリングにスライスする

| 引数 |  |
| :--- | :--- |
| `str` | スライスする [string](https://docs.wandb.ai/ref/weave/string) |
| `begin` | サブストリングの開始インデックス |
| `end` | サブストリングの終了インデックス |

#### 戻り値
サブストリング

<h3 id="string-split"><code>string-split</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) を _リスト_ の [strings](https://docs.wandb.ai/ref/weave/string) に分割する

| 引数 |  |
| :--- | :--- |
| `str` | 分割する [string](https://docs.wandb.ai/ref/weave/string) |
| `sep` | 分割するセパレーター |

#### 戻り値
[strings](https://docs.wandb.ai/ref/weave/string) の _リスト_

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) がプレフィックスで始まるかどうかをチェックする

| 引数 |  |
| :--- | :--- |
| `str` | チェックする [string](https://docs.wandb.ai/ref/weave/string) |
| `prefix` | チェックするプレフィックス |

#### 戻り値
[string](https://docs.wandb.ai/ref/weave/string) がプレフィックスで始まるかどうか

<h3 id="string-strip"><code>string-strip</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) の両端から空白を取り除く

| 引数 |  |
| :--- | :--- |
| `str` | ストリップする [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
ストリップされた [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-upper"><code>string-upper</code></h3>

[string](https://docs.wandb.ai/ref/weave/string) を大文字に変換する

| 引数 |  |
| :--- | :--- |
| `str` | 大文字に変換する [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
大文字に変換された [string](https://docs.wandb.ai/ref/weave/string)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

2つの [strings](https://docs.wandb.ai/ref/weave/string) 間のレーベンシュタイン距離を計算する

| 引数 |  |
| :--- | :--- |
| `str1` | 最初の [string](https://docs.wandb.ai/ref/weave/string) |
| `str2` | 2番目の [string](https://docs.wandb.ai/ref/weave/string) |

#### 戻り値
2つの [strings](https://docs.wandb.ai/ref/weave/string) 間のレーベンシュタイン距離