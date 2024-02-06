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
            <Translate>Documentation</Translate>
          </div>
          <BigSearchBar />
          <div className={styles.homeDescription}>
            <Translate>
            Weights & Biases는 개발자들이 더 나은 모델을 더 빠르게 구축할 수 있도록 하는 기계학습 플랫폼입니다.
            W&B의 가벼우면서도 상호 운용 가능한 도구를 통해 신속하게 실험을 추적하고,
            데이터셋의 버전 관리와 반복 작업을 하며, 모델 성능을 평가하고, 모델을 재현하고,
            결과를 시각화하며, 회귀를 파악하고, 동료들과 결과를 공유할 수 있습니다.
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
                  개발자 가이드는 W&B에 대한 정보와 사용 방법에 대해 심도 있는 정보를 제공합니다. 
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
                  API 참조 가이드는 W&B API에 대한 기술적 정보를 제공합니다.
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
                  W&B가 처음이신가요? 저희의 퀵스타트를 확인해 보세요!
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
                  저희의 튜토리얼을 통해 효율적인 워크플로우를 위한 실용적인 기술을 배워보세요.
                  </Translate>
                </div>
              </div>
            </Link>
          </div>
          <div className={styles.homeFooterBox}>
            저희의 또 다른 페이지인 https://wandb.ai/site{' '}
            <a href="https://wandb.ai/site">
              https://wandb.ai/site.{' '} 에서 W&B
            플랫폼 관련 최신 업데이트를 확인하고, 엔터프라이즈용 W&B에 대해 알아보고,
            ML 커뮤니티와 소통하는 등 다양한 정보를 얻어보세요.
              <ForwardNextIconBlue className={styles.arrowIconLink} />
            </a>
          </div>
        </div>
      </Layout>
    </>
  );
};

export default Home;
