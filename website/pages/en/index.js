/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');
const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const siteConfig = require(process.cwd() + '/siteConfig.js');


function imgUrl(img) {
  return siteConfig.baseUrl + 'img/' + img;
}

function docUrl(doc, language) {
  return siteConfig.baseUrl + 'docs/' + (language ? language + '/' : '') + doc;
}

function pageUrl(page, language) {
  return siteConfig.baseUrl + (language ? language + '/' : '') + page;
}

class Button extends React.Component {
  render() {
    return (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={this.props.href} target={this.props.target}>
          {this.props.children}
        </a>
      </div>
    );
  }
}

Button.defaultProps = {
  target: '_self',
};

const SplashContainer = props => (
  <div className="homeContainer">
    <div className="homeSplashFade">
      <div className="wrapper homeWrapper">{props.children}</div>
    </div>
  </div>
);

const Logo = props => (
  <div className="projectLogo">
    <img src={props.img_src} />
  </div>
);

const ProjectTitle = props => (
  <h1 className="projectTitle">
    <small>{siteConfig.tagline}</small>
  </h1>
);

const PromoSection = props => (
  <div className="section promoSection">
    <div className="promoRow">
      <div className="pluginRowBlock">{props.children}</div>
    </div>
  </div>
);

class HomeSplash extends React.Component {
  render() {
    let language = this.props.language || '';
    const quickLinksCol = [{title: "Getting Started", 
                desc: "Quickly start tracking training.", 
                link:'install.html',
                img: 'img/documentation.png'}, 
                {title: "Python API Reference",
                 desc: "Customize your analysis.",
                 link: 'configs.html',
                img: 'img/api.svg'},
                {title: "Example Projects", 
                desc: "See what W&B can do, and how.", 
                link:'examples.html',
                img: 'img/example.png'}, 
                {title: "FAQ",
                 desc: "Your questions, already answered.",
                 link: 'faq.html',
                  img: 'img/faq.png'}]
    return (
      <SplashContainer>
        <div className="inner">
          <ProjectTitle />
          <div className="allCards" style={{marginBottom: "40px"}}>
              {quickLinksCol.map((quickLink,index) => (
                <div className="grid-item" key={index}>
                <a href={docUrl(quickLink.link, language)}>
                  <div className="cardDisplay">
                    <img src={pageUrl(quickLink.img)} className="icon"/>
                    <div style={{display: "inline-block", marginLeft: "5%"}}>
                      <h3 className="headers">{quickLink.title}</h3>
                      <p>{quickLink.desc}</p>
                    </div>
                  </div>
                </a>
                </div>
              ))}
           </div>
           <a href="https://app.wandb.ai/wandb/face-emotion?view=default">
            <img src={pageUrl("img/teaser.gif")} />
          </a>
        </div>
      </SplashContainer>
    );
  }
}

const Block = props => (
  <Container
    padding={['bottom', 'top']}
    id={props.id}
    background={props.background}>
    <GridBlock align="center" contents={props.children} layout={props.layout} />
  </Container>
);

const Features = props => (
  <Block layout="fourColumn">
    {[
      {
        content: 'This is the content of my feature',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'top',
        title: 'Feature One',
      },
      {
        content: 'The content of my second feature',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'top',
        title: 'Feature Two',
      },
    ]}
  </Block>
);

const FeatureCallout = props => (
  <div
    className="productShowcaseSection paddingBottom"
    style={{textAlign: 'center'}}>
    <h2>Feature Callout</h2>
    <MarkdownBlock>These are features of this project</MarkdownBlock>
  </div>
);

const LearnHow = props => (
  <Block background="light">
    {[
      {
        content: 'Talk about learning how to use this',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'right',
        title: 'Learn How',
      },
    ]}
  </Block>
);

const TryOut = props => (
  <Block id="try">
    {[
      {
        content: 'Talk about trying this out',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'left',
        title: 'Try it Out',
      },
    ]}
  </Block>
);

const Description = props => (
  <Block background="dark">
    {[
      {
        content: 'This is another description of how this project is useful',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'right',
        title: 'Description',
      },
    ]}
  </Block>
);

const Showcase = props => {
  if ((siteConfig.users || []).length === 0) {
    return null;
  }
  const showcase = siteConfig.users
    .filter(user => {
      return user.pinned;
    })
    .map((user, i) => {
      return (
        <a href={user.infoLink} key={i}>
          <img src={user.image} alt={user.caption} title={user.caption} />
        </a>
      );
    });

  return (
    <div className="productShowcaseSection paddingBottom">
      <h3 className="headers" style={{marginTop: "40px", fontSize: "22px"}}>{"Who's Using Weights and Biases?"}</h3>
      <div className="logos">{showcase}</div>
      <div className="more-users">
        <p>Are you using W&B?</p>
        <a href="mailto:contact@wandb.com" className="button">
          Let us know
        </a>
      </div>
    </div>
  );
};

class Index extends React.Component {
  render() {
    let language = this.props.language || '';

    return (
      <div>
        <HomeSplash language={language} />
        <div className="mainContainer">

          <Showcase language={language} />
        </div>
      </div>
    );
  }
}

module.exports = Index;
