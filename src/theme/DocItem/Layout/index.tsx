import React from 'react';
import Layout from '@theme-original/DocItem/Layout';
import Feedback from '@site/src/components/Feedback';
import Chatbot from '@site/src/components/Chatbot';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <Chatbot />
      <Feedback />
    </>
  );
}
