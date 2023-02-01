import React from 'react';
import Layout from '@theme-original/DocItem/Layout';
import Feedback from '@site/src/components/Feedback';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <Feedback />
    </>
  );
}
