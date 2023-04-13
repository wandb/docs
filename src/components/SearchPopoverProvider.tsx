import React, {createContext, useCallback, useContext, useState} from 'react';

const SearchPopoverContext = createContext<{
  setSearchPopoverCallback: (cb: () => void) => void;
  triggerSearchPopover: () => void;
}>(null);

export const useSearchPopoverProvider = () => {
  return useContext(SearchPopoverContext);
};

export const SearchPopoverProvider = ({children}) => {
  const [searchPopoverCallback, setSearchPopoverCallback] =
    useState<() => void>(null);

  const triggerSearchPopover = useCallback(() => {
    if (searchPopoverCallback) {
      searchPopoverCallback();
    }
  }, [searchPopoverCallback]);

  return (
    <SearchPopoverContext.Provider
      value={{
        setSearchPopoverCallback,
        triggerSearchPopover,
      }}>
      {children}
    </SearchPopoverContext.Provider>
  );
};
