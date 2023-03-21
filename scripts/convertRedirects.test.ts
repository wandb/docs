import {describe, test, expect} from '@jest/globals';
import {
  convert,
  convertNew,
  ensureProperRedirectConversion,
  Redirect,
  ConversionError,
} from './convertRedirects';

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
    expect(convertNew(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should convert redirects with common prefixes`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [{from: `/p-a`, to: `/p-b`}];

    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
    expect(convertNew(redirects)).toStrictEqual(expectedConvertedRedirects);
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
    expect(convertNew(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should keep exact redirects as exceptions to prefix redirects`, () => {
    const redirects: Redirect[] = [
      {from: `/p-a/exception`, to: `/unrelated/exception`},
      {from: `/p-a/1`, to: `/p-b/1`},
      {from: `/p-a/2`, to: `/p-b/2`},
    ];
    const expectedConvertedRedirects: Redirect[] = [
      {from: `/p-a/exception`, to: `/unrelated/exception`, exact: true},
      {from: `/p-a`, to: `/p-b`},
    ];
    expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
    expect(convertNew(redirects)).toStrictEqual(expectedConvertedRedirects);
  });

  test(`should not create unnecessarily specific prefix redirects`, () => {
    // const redirects: Redirect[] = [
    //   {from: `/p-a/1`, to: `/p-b/1`},
    //   {from: `/p-a/2`, to: `/p-b/2`},
    // ];
    // const expectedConvertedRedirects: Redirect[] = [
    //   {from: `/a/1`, to: `/a/2`, exact: true},
    //   {from: `/b/1`, to: `/b/2`, exact: true},
    //   {from: `/c/1`, to: `/c/2`, exact: true},
    // ];
    // expect(convert(redirects)).toStrictEqual(expectedConvertedRedirects);
    // expect(convertNew(redirects)).toStrictEqual(expectedConvertedRedirects);
  });
});

describe(`ensureProperRedirectConversion`, () => {
  test(`should catch unhandled redirects`, () => {
    const oldRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`},
      {from: `/b/1`, to: `/b/2`},
      {from: `/c/1`, to: `/c/2`},
    ];
    const convertedRedirects: Redirect[] = [
      {from: `/a/1`, to: `/a/2`, exact: true},
      {from: `/b/1`, to: `/b/2`, exact: true},
    ];
    const expectedErrors: ConversionError[] = [
      {type: `missing`, oldRedirect: {from: `/c/1`, to: `/c/2`}},
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
});
