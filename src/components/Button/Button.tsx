/**
 * This button component is built to December 2022 specifications from the design team.
 * https://www.figma.com/proto/Z6fObCiWnXEVBTtxCpKT4r/Button-specs?node-id=1%3A206&scaling=min-zoom&page-id=0%3A1&starting-point-node-id=1%3A206
 */

import classNames from 'classnames';
import React, {ReactElement, ReactNode} from 'react';

import * as S from './Button.styles';
import {ButtonIconSide, ButtonSize, ButtonVariant} from './types';

export type ButtonProps = {
  Icon?: ReactNode;
  iconSide?: ButtonIconSide;
  size?: ButtonSize;
  variant?: ButtonVariant;
  children?: ReactElement | string;
  disabled?: boolean;
  fullWidth?: boolean;
  active?: boolean;
  isNightMode?: boolean;

  // These are just passed through
  dataTest?: string;
  style?: Record<string, any>;
  className?: string;
  onClick?(e: React.MouseEvent): void;

  as?: any;
  href?: string;
};

export const Button = ({
  size,
  variant,
  Icon,
  iconSide,
  isNightMode,
  disabled,
  fullWidth,
  active,
  children,
  dataTest,
  style,
  className,
  onClick,
  ...props
}: ButtonProps) => {
  let buttonIconSide: string = iconSide ?? 'left';
  if (!children && !Icon) {
    console.error('Button: requires either text (children) or icon.');
  } else if (children && !Icon) {
    buttonIconSide = 'textonly';
  } else if (!children && Icon) {
    buttonIconSide = 'icononly';
  } else if (fullWidth) {
    console.error('Button: cannot use fullWidth with icon');
  }

  const buttonFullWidth = fullWidth ?? false;
  const buttonSize = size ?? 'medium';
  const buttonVariant = variant ?? 'primary';

  const text = children ? (
    <S.Text fullWidth={buttonFullWidth}>{children}</S.Text>
  ) : null;

  const body =
    buttonIconSide === 'right' ? (
      <>
        {text}
        {Icon}
      </>
    ) : (
      <>
        {Icon}
        {text}
      </>
    );

  const button = (
    <S.Button
      isNightMode={isNightMode}
      data-test={dataTest}
      className={classNames('night-aware', className)}
      variant={buttonVariant}
      size={buttonSize}
      iconSide={buttonIconSide}
      fullWidth={buttonFullWidth}
      active={!!active}
      style={style}
      disabled={!!disabled}
      onClick={onClick}
      {...props}>
      {body}
    </S.Button>
  );

  return button;
};
