import React from 'react';
import {SearchPopoverProvider} from '../components/SearchPopoverProvider';

// Default implementation, that you can customize
export default function Root({children}) {
  return <SearchPopoverProvider>{children}</SearchPopoverProvider>;
}
