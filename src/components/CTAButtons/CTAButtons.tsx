
import React from 'react';
import styled from 'styled-components';

import { Button, ButtonSizes, ButtonVariants } from '../Button';
import IconColabLogo from '../Icons/IconColabLogo';
import IconWBLogo from '../Icons/IconWBLogo';
import IconGithubLogo from '../Icons/IconGithubLogo';

const CTAContainer = styled.div`
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
`;


const StyledWBLogo = styled(IconWBLogo)`
    margin: 0 0 0 4px;
`;

const StyledColabLogo = styled(IconColabLogo)`
    margin: 0 8px;
`;

const StyledGithubLogo = styled(IconGithubLogo)`
    margin: 0 8px;
`;

const buttonProps = {
    size: ButtonSizes.Large,
    variant: ButtonVariants.Secondary,
    target:"_blank",
    rel:"noreferrer",
    as:'a'
}

export const ColabButton = ({colabLink}: {colabLink:string}) => 
    <Button 
        href={colabLink} 
        Icon={<StyledColabLogo height={24} width={24}/>} 
        onClick={() => window.analytics?.track('Docs button clicked', {type: 'Try in colab'})}
        {...buttonProps}>
        Try in colab
    </Button>

export const ProductButton = ({productLink}: {productLink:string}) =>
    <Button 
        href={productLink} 
        Icon={<StyledWBLogo height={35} width={35}/>} 
        onClick={() => window.analytics?.track('Docs button clicked', {type: 'Try in product'})}
        {...buttonProps}>
        Try in product
    </Button>

export const GithubButton = ({githubLink}: {githubLink:string}) =>
    <Button 
        href={githubLink} 
        Icon={<StyledGithubLogo height={24} width={24}/>} 
        onClick={() => window.analytics?.track('Docs button clicked', {type: 'Try the code'})}
        {...buttonProps}>
        Try the code
    </Button>

export const CTAButtons = ({productLink, colabLink, githubLink}: {productLink?:string, colabLink?:string, githubLink?:string}) => 
    <CTAContainer>
        {productLink && <ProductButton productLink={productLink}/>}
        {colabLink && <ColabButton colabLink={colabLink}/> }
        {githubLink && <GithubButton githubLink={githubLink}/> }
    </CTAContainer>