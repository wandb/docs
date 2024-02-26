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

import Translate, {translate} from '@docusaurus/Translate';

const BigSearchBar = () => {
  const {triggerSearchPopover} = useSearchPopoverProvider();
  const onClick = useCallback(() => {
    triggerSearchPopover();
  }, [triggerSearchPopover]);

  return (
    <div className={styles.bigSearch} onClick={onClick}>
      <SearchIcon className={styles.searchIcon} />
      <Translate>Search documentation</Translate>
    </div>
  );
};

const Home: React.FC = () => {
  return (
    <>
      <Layout title="W&B Docs">
        <div className={styles.homePage}>
          <div className={styles.homeHeader}>
            <Translate>Weights & Biases</Translate> <br/>
            <Translate>ドキュメンテーション</Translate>
          </div>
          <BigSearchBar />
          <div className={styles.homeDescription}>
            <Translate>
            Weights & Biasesは機械学習開発に関わる全ての人たちが、より良いモデルをより早く構築することを支援します
            W&Bは軽量で、互換性が高く、実験管理、データセットのバージョン管理、モデルの品質評価、再現可能性の向上、
            結果の可視化、チームとのコラボレーションなどの優れたワークフローをあなたの開発プロセスにすぐ導入することができます。
            </Translate>
          </div>
          <div className={styles.homeBoxContainer}>
            <Link to="/guides">
              <div className={clsx(styles.homeBox, styles.yellow)}>
                <QuickStartIcon />
                <div className={styles.boxHeader}>
                <Translate>Developer guide</Translate>
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  <Translate>
                  W&Bの使い方および開発者向けの詳しい情報を提供します。 
                  </Translate>
                </div>
              </div>
            </Link>
            <Link to="/ref">
              <div className={clsx(styles.homeBox, styles.sienna)}>
                <CodeAltIcon />
                <div className={styles.boxHeader}>
                  <Translate>API Reference</Translate>
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  <Translate>
                  W&BのAPIに関する技術的な情報を提供します
                  </Translate>
                </div>
              </div>
            </Link>
            <Link to="/quickstart">
              <div className={clsx(styles.homeBox, styles.gray)}>
                <MagicWandStarIcon />
                <div className={styles.boxHeader}>
                  <Translate>Quickstart</Translate>
                  <ForwardNextIcon className={styles.arrowIcon} />
                </div>
                <div className={styles.boxDescription}>
                  <Translate>
                  とりあえず初めてみたい、という方は、このクイックスタートから
                  </Translate>
                </div>
              </div>
            </Link>
            <Link to="/tutorials">
              <div className={clsx(styles.homeBox, styles.teal)}>
                <DocumentationIcon />
                <div className={clsx(styles.boxHeader, styles.whiteText)}>
                  <Translate>Tutorials</Translate>
                  <ForwardNextIconWhite className={styles.arrowIcon} />
                </div>
                <div className={clsx(styles.boxDescription, styles.whiteText)}>
                  <Translate>
                  実世界の課題をテーマにしたチュートリアルの一覧です（英語)
                  </Translate>
                </div>
              </div>
            </Link>
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
