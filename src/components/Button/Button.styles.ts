import styled, {css} from 'styled-components';

import {
  getBackgroundColor,
  getBackgroundColorHover,
  getColor,
  getColorDisabled,
  getColorHover,
  getFontSize,
  getHeight,
  getLineHeight,
  getPadding,
  getPaddingLeft,
  getPaddingRight,
} from './utils';

type TextProps = {
  fullWidth: boolean;
};

export const Text = styled.div<TextProps>`
  margin: ${props => (props.fullWidth ? 'auto' : undefined)};
`;

type ButtonProps = {
  isNightMode: boolean;
  fullWidth: boolean;
  size: string;
  variant: string;
  iconSide: string;
  disabled: boolean;
  active: boolean;
};

export const Button = styled.button<ButtonProps>`
  border: 0;
  border-radius: 4px;
  font-weight: 600;
  font-family: 'Source Sans Pro';
  display: inline-flex;
  align-items: center;
  padding: ${props => getPadding(props.size, props.iconSide)};
  padding-left: ${props => getPaddingLeft(props.size, props.iconSide)};
  padding-right: ${props => getPaddingRight(props.size, props.iconSide)};
  cursor: ${props => (props.disabled ? 'default' : 'pointer')};
  pointer-events: ${props => (props.disabled ? 'none' : 'auto')};
  white-space: nowrap;

  height: ${props => getHeight(props.size)};
  line-height: ${props => getLineHeight(props.size)};
  font-size: ${props => getFontSize(props.size)};

  color: ${props => getColor(props.variant, props.isNightMode)};
  background-color: ${props =>
    getBackgroundColor(props.variant, props.isNightMode, props.disabled)};

  width: ${props => (props.fullWidth ? '100%' : undefined)};

  &:disabled {
    color: ${props => getColorDisabled(props.isNightMode)};
  }

  // Apply styling when button needs to remain selected/active
  ${props =>
    props.active &&
    css`
      color: ${getColorHover(props.variant, props.isNightMode, props.disabled)};
      background-color: ${getBackgroundColorHover(
        props.variant,
        props.isNightMode,
        props.disabled
      )};
    `}

  &:hover {
    color: ${props =>
      getColorHover(props.variant, props.isNightMode, props.disabled)};
    background-color: ${props =>
      getBackgroundColorHover(
        props.variant,
        props.isNightMode,
        props.disabled
      )};
  }
`;
