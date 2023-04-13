import React, {useCallback} from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

import styles from './styles.module.css';

import CodeAltIcon from '/static/img/icon-code-alt.svg';
import DocumentationIcon from '/static/img/icon-documentation.svg';
import MagicWandStarIcon from '/static/img/icon-magic-wand-star.svg';
import QuickStartIcon from '/static/img/icon-quickstart-code.svg';
import ForwardNextIcon from '/static/img/icon-forward-next-gray-800.svg';
import ForwardNextIconBlue from '/static/img/icon-forward-next.svg';
import ForwardNextIconWhite from '/static/img/icon-forward-next-white.svg';
import SearchIcon from '/static/img/icon-search-grey.svg';
import {useSearchPopoverProvider} from '../components/SearchPopoverProvider';
import clsx from 'clsx';

const BigSearchBar = () => {
  const {triggerSearchPopover} = useSearchPopoverProvider();
  const onClick = useCallback(() => {
    triggerSearchPopover();
  }, [triggerSearchPopover]);

  return (
    <div className={styles.bigSearch} onClick={onClick}>
      <SearchIcon className={styles.searchIcon} />
      Search documentation
    </div>
  );
};

const Home: React.FC = () => {
  return (
    <>
      <Layout title="W&B Docs">
        <div className={styles.homePage}>
          <div className={styles.homeHeader}>
            Weights & Biases Documentation
          </div>
          <BigSearchBar />
          <div className={styles.homeDescription}>
            Weights & Biases is the machine learning platform for developers to
            build better models faster. Use W&B's lightweight, interoperable
            tools to quickly track experiments, version and iterate on datasets,
            evaluate model performance, reproduce models, visualize results and
            spot regressions, and share findings with colleagues.
          </div>
          <div className={styles.homeBoxContainer}>
            <Link to="/guide">
              <div className={clsx(styles.homeBox, styles.yellow)}>
                <QuickStartIcon />
                <div className={styles.boxHeader}>
                  Developer Guide
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  The developer guide provides in-depth information about W&B
                  and how to use it.
                </div>
              </div>
            </Link>
            <Link to="/ref">
              <div className={clsx(styles.homeBox, styles.sienna)}>
                <CodeAltIcon />
                <div className={styles.boxHeader}>
                  API Reference
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  The API reference guide provides technical information about
                  the W&B API.
                </div>
              </div>
            </Link>
            <Link to="/quickstart">
              <div className={clsx(styles.homeBox, styles.gray)}>
                <MagicWandStarIcon />
                <div className={styles.boxHeader}>
                  Tutorials
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  Are you new to W&B? Check out our tutorials!
                </div>
              </div>
            </Link>
            <a href="https://wandb.ai/site/tutorials">
              <div className={clsx(styles.homeBox, styles.teal)}>
                <DocumentationIcon />
                <div className={clsx(styles.boxHeader, styles.whiteText)}>
                  How-to guides{' '}
                  <ForwardNextIconWhite className={styles.arrowIcon} />
                </div>
                <div className={clsx(styles.boxDescription, styles.whiteText)}>
                  Explore examples to real-world problems with our how-to
                  guides.
                </div>
              </div>
            </a>
          </div>
          <div className={styles.homeFooterBox}>
            Stay up to date with the latest updates from our W&B platform, learn
            about W&B for enterprise, stay connected with the ML community and
            more at our sister page:{' '}
            <a href="https://wandb.ai/site">
              https://wandb.ai/site.{' '}
              <ForwardNextIconBlue className={styles.arrowIconLink} />
            </a>
          </div>
        </div>
      </Layout>
    </>
  );
};

export default Home;
