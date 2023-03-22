import _ from 'lodash';
import {log, stringify} from '../utils';

export type Redirect = {
  from: string;
  to: string;

  // This is the same `exact` prop used by `react-router`. When `exact !== true`,
  // the redirect will be applied to all paths that start with `from`.
  exact?: boolean;
};

export function convert(redirects: Redirect[]): Redirect[] {
  const {withoutAbsolute, withAbsolute} = groupRedirectsByAbsolute(redirects);

  const inexactRedirectsWithIndices =
    getInexactRedirectsWithIndices(withoutAbsolute);
  const inexactRedirects: Redirect[] = [];
  const inexactIndices = new Set<number>();
  for (const {redirect, indices} of inexactRedirectsWithIndices) {
    inexactRedirects.push(redirect);
    for (const index of indices) {
      inexactIndices.add(index);
    }
  }

  const exactRedirects = withoutAbsolute.filter(
    (r, i) => !inexactIndices.has(i)
  );

  return [
    ...sortExactRedirects(addExactProp([...exactRedirects, ...withAbsolute])),
    ...sortInexactRedirects(inexactRedirects),
  ];
}

function sortExactRedirects(redirects: Redirect[]): Redirect[] {
  return _.sortBy(redirects, r => r.from);
}

function sortInexactRedirects(redirects: Redirect[]): Redirect[] {
  return _.sortBy(
    _.sortBy(redirects, r => r.from),

    // Longer paths should take precedence over shorter paths
    r => -r.from.split('/').length
  );
}

function addExactProp(redirects: Redirect[]): Redirect[] {
  return redirects.map(r => ({...r, exact: true}));
}

type RedirectData = {
  fromPrefix: string;
  toPrefix: string;
  suffix: string;
  index: number;
};

type RedirectWithIndices = {
  redirect: Redirect;
  indices: number[];
};

function getInexactRedirectsWithIndices(
  redirects: Redirect[]
): RedirectWithIndices[] {
  const inexactRedirects: RedirectWithIndices[] = [];
  const handledIndices = new Set<number>();
  const maxFromSegmentCount = getMaxFromSegmentCount(redirects);
  for (
    let fromSegmentCount = 1;
    fromSegmentCount < maxFromSegmentCount;
    fromSegmentCount++
  ) {
    const datas: RedirectData[] = redirects
      .map(getRedirectData)
      .filter(isNotNullOrUndefined);

    Object.values(_.groupBy(datas, d => d.fromPrefix))
      .filter(moreThanOneRedirect)
      .filter(allToPrefixesEqual)
      .forEach(datas => {
        const filteredIndices = datas
          .map(d => d.index)
          .filter(i => !handledIndices.has(i));
        if (filteredIndices.length === 0) {
          return;
        }

        inexactRedirects.push({
          redirect: {
            from: datas[0]!.fromPrefix,
            to: datas[0]!.toPrefix,
          },
          indices: filteredIndices,
        });
        for (const index of filteredIndices) {
          handledIndices.add(index);
        }
      });

    function getRedirectData(r: Redirect, i: number): RedirectData | null {
      const fromPrefix = truncateToNSegments(r.from, fromSegmentCount);
      const fromSuffix = r.from.slice(fromPrefix.length);
      if (!r.to.endsWith(fromSuffix)) {
        // Exclude redirects where from and to have different suffixes
        return null;
      }
      const toPrefix = r.to.slice(0, r.to.length - fromSuffix.length);
      return {
        fromPrefix,
        toPrefix,
        suffix: fromSuffix,
        index: i,
      };
    }

    function moreThanOneRedirect(datas: RedirectData[]): boolean {
      return datas.length > 1;
    }
    function allToPrefixesEqual(datas: RedirectData[]): boolean {
      return datas.every(({toPrefix}) => toPrefix === datas[0]!.toPrefix);
    }
  }

  return inexactRedirects;
}

function getSegmentsFromPath(path: string): string[] {
  return killLeadingSlash(path).split('/');
}

function getMaxFromSegmentCount(redirects: Redirect[]): number {
  const fromSegmentCounts = redirects.map(
    r => getSegmentsFromPath(r.from).length
  );
  return Math.max(...fromSegmentCounts);
}

function truncateToNSegments(path: string, n: number): string {
  const segments = getSegmentsFromPath(path);
  return `/${segments.slice(0, n).join('/')}`;
}

function killLeadingSlash(path: string): string {
  if (path.startsWith(`/`)) {
    return path.slice(1);
  }
  return path;
}

function isNotNullOrUndefined<T>(x: T | null | undefined): x is T {
  return x != null;
}

function groupRedirectsByAbsolute(redirects: Redirect[]): {
  withAbsolute: Redirect[];
  withoutAbsolute: Redirect[];
} {
  const withAbsolute: Redirect[] = [];
  const withoutAbsolute: Redirect[] = [];
  for (const redirect of redirects) {
    if (redirect.from.startsWith(`http`) || redirect.to.startsWith(`http`)) {
      withAbsolute.push(redirect);
    } else {
      withoutAbsolute.push(redirect);
    }
  }
  return {withAbsolute, withoutAbsolute};
}

export type ConversionError = MissingRedirectError | WrongRedirectError;

type MissingRedirectError = {
  type: 'missing';
  oldRedirect: Redirect;
};

type WrongRedirectError = {
  type: 'wrong';
  oldRedirect: Redirect;
  convertedRedirect: Redirect;
};

export function ensureProperRedirectConversion(
  ogRedirects: Redirect[],
  convertedRedirects: Redirect[]
): ConversionError[] {
  const errors: ConversionError[] = [];
  OuterLoop: for (const oldRedirect of ogRedirects) {
    const appliedRedirect = getAppliedRedirect(
      convertedRedirects,
      oldRedirect.from
    );
    if (appliedRedirect == null) {
      pushMissing();
      continue;
    }

    if (appliedRedirect.exact) {
      if (oldRedirect.to !== appliedRedirect.to) {
        pushWrong(appliedRedirect);
      }
      continue;
    }

    const appliedTo = `${appliedRedirect.to}${getRedirectSuffix(
      appliedRedirect.from,
      oldRedirect.from
    )}`;

    if (oldRedirect.to !== appliedTo) {
      pushWrong(appliedRedirect);
    }

    function pushMissing(): void {
      errors.push({
        type: 'missing',
        oldRedirect: oldRedirect,
      });
    }
    function pushWrong(convertedRedirect: Redirect) {
      errors.push({
        type: 'wrong',
        oldRedirect: oldRedirect,
        convertedRedirect,
      });
    }
  }

  return errors;
}

function getAppliedRedirect(
  redirects: Redirect[],
  path: string
): Redirect | null {
  for (const redirect of redirects) {
    if (redirect.exact && redirect.from === path) {
      return redirect;
    }
    if (!redirect.exact && isPrefix(redirect.from, path)) {
      return redirect;
    }
  }
  return null;
}

function getRedirectSuffix(prefix: string, path: string): string | null {
  if (!isPrefix(prefix, path)) {
    return null;
  }
  return path.slice(prefix.length);
}

function isPrefix(prefix: string, path: string): boolean {
  const prefixSegments = getSegmentsFromPath(prefix);
  const pathSegments = getSegmentsFromPath(path);
  for (let i = 0; i < prefixSegments.length; i++) {
    if (prefixSegments[i] !== pathSegments[i]) {
      return false;
    }
  }
  return true;
}

export function logConversionErrors(errors: ConversionError[]): void {
  for (const error of errors) {
    switch (error.type) {
      case 'missing':
        log(`Missing redirect: ${stringify(error.oldRedirect)}`);
        break;
      case 'wrong':
        log(
          `Wrong redirect: ${stringify(
            error.oldRedirect
          )} converted to ${stringify(error.convertedRedirect)}`
        );
        break;
    }
  }
}
