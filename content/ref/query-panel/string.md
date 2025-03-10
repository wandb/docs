---
title: string
---
## Chainable Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

Determines inequality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are not equal.

<h3 id="string-add"><code>string-add</code></h3>

Concatenates two [strings](string.md)

| Argument |  |
| :--- | :--- |
| `lhs` | The first [string](string.md) |
| `rhs` | The second [string](string.md) |

#### Return Value
The concatenated [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

Determines equality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are equal.

<h3 id="string-append"><code>string-append</code></h3>

Appends a suffix to a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to append to |
| `suffix` | The suffix to append |

#### Return Value
The [string](string.md) with the suffix appended

<h3 id="string-contains"><code>string-contains</code></h3>

Checks if a [string](string.md) contains a substring

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `sub` | The substring to check for |

#### Return Value
Whether the [string](string.md) contains the substring

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

Checks if a [string](string.md) ends with a suffix

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `suffix` | The suffix to check for |

#### Return Value
Whether the [string](string.md) ends with the suffix

<h3 id="string-findAll"><code>string-findAll</code></h3>

Finds all occurrences of a substring in a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to find occurrences of the substring in |
| `sub` | The substring to find |

#### Return Value
The _list_ of indices of the substring in the [string](string.md)

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

Checks if a [string](string.md) is alphanumeric

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is alphanumeric

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

Checks if a [string](string.md) is alphabetic

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is alphabetic

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

Checks if a [string](string.md) is numeric

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is numeric

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

Strip leading whitespace

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-len"><code>string-len</code></h3>

Returns the length of a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
The length of the [string](string.md)

<h3 id="string-lower"><code>string-lower</code></h3>

Converts a [string](string.md) to lowercase

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to convert to lowercase |

#### Return Value
The lowercase [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

Partitions a [string](string.md) into a _list_ of the [strings](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to split |
| `sep` | The separator to split on |

#### Return Value
A _list_ of [strings](string.md): the [string](string.md) before the separator, the separator, and the [string](string.md) after the separator

<h3 id="string-prepend"><code>string-prepend</code></h3>

Prepends a prefix to a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to prepend to |
| `prefix` | The prefix to prepend |

#### Return Value
The [string](string.md) with the prefix prepended

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

Strip trailing whitespace

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-replace"><code>string-replace</code></h3>

Replaces all occurrences of a substring in a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to replace contents of |
| `sub` | The substring to replace |
| `newSub` | The substring to replace the old substring with |

#### Return Value
The [string](string.md) with the replacements

<h3 id="string-slice"><code>string-slice</code></h3>

Slices a [string](string.md) into a substring based on beginning and end indices

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to slice |
| `begin` | The beginning index of the substring |
| `end` | The ending index of the substring |

#### Return Value
The substring

<h3 id="string-split"><code>string-split</code></h3>

Splits a [string](string.md) into a _list_ of [strings](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to split |
| `sep` | The separator to split on |

#### Return Value
The _list_ of [strings](string.md)

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

Checks if a [string](string.md) starts with a prefix

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `prefix` | The prefix to check for |

#### Return Value
Whether the [string](string.md) starts with the prefix

<h3 id="string-strip"><code>string-strip</code></h3>

Strip whitespace from both ends of a [string](string.md).

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-upper"><code>string-upper</code></h3>

Converts a [string](string.md) to uppercase

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to convert to uppercase |

#### Return Value
The uppercase [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

Calculates the Levenshtein distance between two [strings](string.md).

| Argument |  |
| :--- | :--- |
| `str1` | The first [string](string.md). |
| `str2` | The second [string](string.md). |

#### Return Value
The Levenshtein distance between the two [strings](string.md).


## List Ops
<h3 id="string-notEqual"><code>string-notEqual</code></h3>

Determines inequality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are not equal.

<h3 id="string-add"><code>string-add</code></h3>

Concatenates two [strings](string.md)

| Argument |  |
| :--- | :--- |
| `lhs` | The first [string](string.md) |
| `rhs` | The second [string](string.md) |

#### Return Value
The concatenated [string](string.md)

<h3 id="string-equal"><code>string-equal</code></h3>

Determines equality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are equal.

<h3 id="string-append"><code>string-append</code></h3>

Appends a suffix to a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to append to |
| `suffix` | The suffix to append |

#### Return Value
The [string](string.md) with the suffix appended

<h3 id="string-contains"><code>string-contains</code></h3>

Checks if a [string](string.md) contains a substring

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `sub` | The substring to check for |

#### Return Value
Whether the [string](string.md) contains the substring

<h3 id="string-endsWith"><code>string-endsWith</code></h3>

Checks if a [string](string.md) ends with a suffix

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `suffix` | The suffix to check for |

#### Return Value
Whether the [string](string.md) ends with the suffix

<h3 id="string-findAll"><code>string-findAll</code></h3>

Finds all occurrences of a substring in a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to find occurrences of the substring in |
| `sub` | The substring to find |

#### Return Value
The _list_ of indices of the substring in the [string](string.md)

<h3 id="string-isAlnum"><code>string-isAlnum</code></h3>

Checks if a [string](string.md) is alphanumeric

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is alphanumeric

<h3 id="string-isAlpha"><code>string-isAlpha</code></h3>

Checks if a [string](string.md) is alphabetic

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is alphabetic

<h3 id="string-isNumeric"><code>string-isNumeric</code></h3>

Checks if a [string](string.md) is numeric

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
Whether the [string](string.md) is numeric

<h3 id="string-lStrip"><code>string-lStrip</code></h3>

Strip leading whitespace

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-len"><code>string-len</code></h3>

Returns the length of a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |

#### Return Value
The length of the [string](string.md)

<h3 id="string-lower"><code>string-lower</code></h3>

Converts a [string](string.md) to lowercase

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to convert to lowercase |

#### Return Value
The lowercase [string](string.md)

<h3 id="string-partition"><code>string-partition</code></h3>

Partitions a [string](string.md) into a _list_ of the [strings](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to split |
| `sep` | The separator to split on |

#### Return Value
A _list_ of [strings](string.md): the [string](string.md) before the separator, the separator, and the [string](string.md) after the separator

<h3 id="string-prepend"><code>string-prepend</code></h3>

Prepends a prefix to a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to prepend to |
| `prefix` | The prefix to prepend |

#### Return Value
The [string](string.md) with the prefix prepended

<h3 id="string-rStrip"><code>string-rStrip</code></h3>

Strip trailing whitespace

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-replace"><code>string-replace</code></h3>

Replaces all occurrences of a substring in a [string](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to replace contents of |
| `sub` | The substring to replace |
| `newSub` | The substring to replace the old substring with |

#### Return Value
The [string](string.md) with the replacements

<h3 id="string-slice"><code>string-slice</code></h3>

Slices a [string](string.md) into a substring based on beginning and end indices

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to slice |
| `begin` | The beginning index of the substring |
| `end` | The ending index of the substring |

#### Return Value
The substring

<h3 id="string-split"><code>string-split</code></h3>

Splits a [string](string.md) into a _list_ of [strings](string.md)

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to split |
| `sep` | The separator to split on |

#### Return Value
The _list_ of [strings](string.md)

<h3 id="string-startsWith"><code>string-startsWith</code></h3>

Checks if a [string](string.md) starts with a prefix

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to check |
| `prefix` | The prefix to check for |

#### Return Value
Whether the [string](string.md) starts with the prefix

<h3 id="string-strip"><code>string-strip</code></h3>

Strip whitespace from both ends of a [string](string.md).

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to strip. |

#### Return Value
The stripped [string](string.md).

<h3 id="string-upper"><code>string-upper</code></h3>

Converts a [string](string.md) to uppercase

| Argument |  |
| :--- | :--- |
| `str` | The [string](string.md) to convert to uppercase |

#### Return Value
The uppercase [string](string.md)

<h3 id="string-levenshtein"><code>string-levenshtein</code></h3>

Calculates the Levenshtein distance between two [strings](string.md).

| Argument |  |
| :--- | :--- |
| `str1` | The first [string](string.md). |
| `str2` | The second [string](string.md). |

#### Return Value
The Levenshtein distance between the two [strings](string.md).

