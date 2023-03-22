import {describe, test, expect} from '@jest/globals';
import {
  convert,
  ensureProperRedirectConversion,
  Redirect,
  ConversionError,
} from './lib';

describe(`convert`, () => {
  test(`should not convert redirects without common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/c/1`, to: `/c/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`, exact: true},
      {from: `/b/1`, to: `/b/2`, exact: true},
      {from: `/c/1`, to: `/c/2`, exact: true},
    ];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should convert redirects with common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [{from: `/p-a`, to: `/p-b`}];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should not convert exact redirects unrelated to prefix redirects`, () => {
    const redirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`, exact: true},
      {from: `/b/1`, to: `/b/2`, exact: true},
      {from: `/p-a`, to: `/p-b`},
    ];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should keep exact redirects as exceptions to prefix redirects`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/exception`, to: `/unrelated/redirect`},
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [
      {from: `/p-a/exception`, to: `/unrelated/redirect`, exact: true},
      {from: `/p-a`, to: `/p-b`},
    ];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should create separate prefix redirects when necessary`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/a/1`, to: `/p-b/a/1`},
      {from: `/p-a/a/2`, to: `/p-b/a/2`},
      {from: `/p-a/b/1`, to: `/p-b/1`},
      {from: `/p-a/b/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [
      {from: `/p-a/b`, to: `/p-b`},
      {from: `/p-a`, to: `/p-b`},
    ];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should not create unnecessarily specific prefix redirects`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/a/b/c/1`, to: `/p-b/a/b/c/1`},
      {from: `/p-a/a/b/2`, to: `/p-b/a/b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [{from: `/p-a`, to: `/p-b`}];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
  });
});

describe(`ensureProperRedirectConversion`, () => {
  test(`should catch unhandled redirects`, () => {
    const oldRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/c/1`, to: `/c/2`},

      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
      {from: `/p-c/1`, to: `/p-d/1`},
      {from: `/p-c/2`, to: `/p-d/2`},
    ];
    const convertedRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`, exact: true},
      {from: `/b/1`, to: `/b/2`, exact: true},

      {from: `/p-a`, to: `/p-b`},
    ];
    const expectedErrors: ConversionError[] = [
      {type: `missing`, oldRedirect: {from: `/c/1`, to: `/c/2`}},
      {type: `missing`, oldRedirect: {from: `/p-c/1`, to: `/p-d/1`}},
      {type: `missing`, oldRedirect: {from: `/p-c/2`, to: `/p-d/2`}},
    ];

    expect(
      ensureProperRedirectConversion(oldRedirects, convertedRedirects)
    ).toStrictEqual(expectedErrors);
  });

  test(`should catch wrong exact redirects`, () => {
    const oldRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/c/1`, to: `/c/2`},
    ];
    const convertedRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`, exact: true},
      {from: `/b/1`, to: `/b/2`, exact: true},
      {from: `/c/1`, to: `/d/2`, exact: true},
    ];
    const expectedErrors: ConversionError[] = [
      {
        type: `wrong`,
        oldRedirect: {from: `/c/1`, to: `/c/2`},
        convertedRedirect: {from: `/c/1`, to: `/d/2`, exact: true},
      },
    ];
    expect(
      ensureProperRedirectConversion(oldRedirects, convertedRedirects)
    ).toStrictEqual(expectedErrors);
  });

  test(`should catch wrong inexact redirects`, () => {
    const oldRedirects: Redirect[] = [
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
      {from: `/p-c/1`, to: `/p-d/1`},
      {from: `/p-c/2`, to: `/p-d/2`},
    ];
    const convertedRedirects: Redirect[] = [
      {from: `/p-a`, to: `/p-b`},
      {from: `/p-c`, to: `/p-e`},
    ];
    const expectedErrors: ConversionError[] = [
      {
        type: `wrong`,
        oldRedirect: {from: `/p-c/1`, to: `/p-d/1`},
        convertedRedirect: {from: `/p-c`, to: `/p-e`},
      },
      {
        type: `wrong`,
        oldRedirect: {from: `/p-c/2`, to: `/p-d/2`},
        convertedRedirect: {from: `/p-c`, to: `/p-e`},
      },
    ];
    expect(
      ensureProperRedirectConversion(oldRedirects, convertedRedirects)
    ).toStrictEqual(expectedErrors);
  });
});
