import {describe, test, expect} from '@jest/globals';
import {getSuggestionPrefixes} from './lib';
import type {Redirect} from '../utils';

describe(`getSuggestionPrefixes`, () => {
  test(`should get no suggestions from redirects without common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/c/1`, to: `/c/2`},
    ];
    const suggestionPrefixes: Redirect[] = [];
    expect(getSuggestionPrefixes(redirects)).toStrictEqual(suggestionPrefixes);
  });

  test(`should get suggestions from redirects with common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
      {from: `/p-c/1`, to: `/p-b/1`},
      {from: `/p-c/2`, to: `/p-b/2`},
    ];
    const suggestionPrefixes: Redirect[] = [
      {from: `/p-a`, to: `/p-b`},
      {from: `/p-c`, to: `/p-b`},
    ];
    expect(getSuggestionPrefixes(redirects)).toStrictEqual(suggestionPrefixes);
  });

  test(`should ignore redirects without common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/p-a/exception`, to: `/unrelated/redirect`},
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const suggestionPrefixes: Redirect[] = [{from: `/p-a`, to: `/p-b`}];
    expect(getSuggestionPrefixes(redirects)).toStrictEqual(suggestionPrefixes);
  });

  test(`should get separate suggestions when necessary`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/a/1`, to: `/p-b/a/1`},
      {from: `/p-a/a/2`, to: `/p-b/a/2`},
      {from: `/p-a/b/1`, to: `/p-b/1`},
      {from: `/p-a/b/2`, to: `/p-b/2`},
    ];
    const suggestionPrefixes: Redirect[] = [
      {from: `/p-a/b`, to: `/p-b`},
      {from: `/p-a`, to: `/p-b`},
    ];
    expect(getSuggestionPrefixes(redirects)).toStrictEqual(suggestionPrefixes);
  });

  test(`should not get unnecessarily specific suggestions`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/a/b/c/1`, to: `/p-b/a/b/c/1`},
      {from: `/p-a/a/b/2`, to: `/p-b/a/b/2`},
    ];
    const suggestionPrefixes: Redirect[] = [{from: `/p-a`, to: `/p-b`}];
    expect(getSuggestionPrefixes(redirects)).toStrictEqual(suggestionPrefixes);
  });
});
