import React from 'react';
import clsx from 'clsx';
import TOCItems from '@theme/TOCItems';
import styles from './styles.module.css';
// Using a custom className
// This prevents TOCInline/TOCCollapsible getting highlighted by mistake
const LINK_CLASS_NAME = 'table-of-contents__link toc-highlight';
const LINK_ACTIVE_CLASS_NAME = 'table-of-contents__link--active';
export default function TOC({className, ...props}) {
  return (
    <div className={clsx(styles.tableOfContents, 'thin-scrollbar', className)}>
      {/* 
        the only thing added in this swizzle was this header
        Header should only appear if we are goign to display a TOC
        TOC only appears for items with a level less that 4
        Fixes a bug where header would appear but toc would render
      */}
      {props.toc.filter( ({level}) => level < 4).length > 0 && <span className='TOC-header'>On this page</span>}
      <TOCItems
        {...props}
        linkClassName={LINK_CLASS_NAME}
        linkActiveClassName={LINK_ACTIVE_CLASS_NAME}
      />
    </div>
);
}
