
import React from 'react';
import styled from 'styled-components';

import { Button, ButtonSizes, ButtonVariants } from '../Button';
import IconColabLogo from '../Icons/IconColabLogo';
import IconWBLogo from '../Icons/IconWBLogo';

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

export const CTAButtons = ({productLink, colabLink}: {productLink:string, colabLink:string}) => {
    const buttonProps = {
        size: ButtonSizes.Large,
        variant: ButtonVariants.Secondary,
        target:"_blank",
        rel:"noreferrer",
        as:'a'
    }

    return (
        <CTAContainer>
            <Button href={productLink} Icon={<StyledWBLogo height={35} width={35}/>} onClick={() => {console.log("something"); window.analytics?.track('Docs button clicked', {type: 'Try in product'})}} {...buttonProps}>Try in product</Button>
            <Button href={colabLink} Icon={<StyledColabLogo height={24} width={24}/>} onClick={() => {console.log("something"); window.analytics?.track('Docs button clicked', {type: 'Try in colab'})}}{...buttonProps}>Try in colab</Button>
        </CTAContainer>
    );
};