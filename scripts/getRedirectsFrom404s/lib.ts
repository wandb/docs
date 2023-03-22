import _ from 'lodash';
import {isNotNullOrUndefined} from '../utils';
import {
  getMaxFromSegmentCount,
  isRelativeRedirect,
  sortSuggestionPrefixes,
  truncateToNSegments,
} from './utils';

export type Redirect = {
  from: string;
  to: string;
};

export function getSuggestionPrefixes(redirects: Redirect[]): Redirect[] {
  const relativeRedirects = redirects.filter(isRelativeRedirect);
  return getSuggestionPrefixesFromRelativeRedirects(relativeRedirects);
}

type RedirectData = {
  fromPrefix: string;
  toPrefix: string;
  suffix: string;
  index: number;
};

function getSuggestionPrefixesFromRelativeRedirects(
  redirects: Redirect[]
): Redirect[] {
  const suggestionPrefixes: Redirect[] = [];
  const addedIndices = new Set<number>();
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
        const unaddedIndices = datas
          .map(d => d.index)
          .filter(i => !addedIndices.has(i));
        if (unaddedIndices.length === 0) {
          return;
        }

        suggestionPrefixes.push({
          from: datas[0]!.fromPrefix,
          to: datas[0]!.toPrefix,
        });
        for (const index of unaddedIndices) {
          addedIndices.add(index);
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

  return sortSuggestionPrefixes(suggestionPrefixes);
}
