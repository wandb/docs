import {Colors, hexToRGB} from '../../css/globals.styles'

export const getHeight = (size: string) => {
  switch (size) {
    case 'small':
      return '24px';
    case 'medium':
      return '32px';
    case 'large':
      return '40px';
  }
  return undefined;
};

export const getLineHeight = (size: string) => {
  switch (size) {
    case 'small':
      return '18px';
    case 'medium':
    case 'large':
      return '20px';
  }
  return undefined;
};

export const getFontSize = (size: string) => {
  switch (size) {
    case 'small':
      return '14px';
    case 'medium':
    case 'large':
      return '16px';
  }
  return undefined;
};

export const getColor = (color: string, isNightMode: boolean) => {
  switch (color) {
    case 'primary':
    case 'destructive':
      return Colors.WHITE;
    case 'secondary':
    case 'ghost':
      return isNightMode ? Colors.WHITE : Colors.GRAY_800;
  }
  return undefined;
};

export const getColorDisabled = (isNightMode: boolean) => {
  const clr = isNightMode ? Colors.WHITE : Colors.GRAY_800;
  return hexToRGB(clr, 0.32);
};

export const getBackgroundColor = (
  color: string,
  isNightMode: boolean,
  disabled: boolean
) => {
  if (disabled) {
    color = 'secondary';
  }
  switch (color) {
    case 'primary':
      return Colors.TEAL_LIGHT;
    case 'secondary':
      return isNightMode
        ? hexToRGB(Colors.WHITE, 0.12)
        : hexToRGB(Colors.GRAY_900, 0.05);
    case 'ghost':
      return Colors.TRANSPARENT;
    case 'destructive':
      return Colors.RED;
  }
  return undefined;
};

export const getColorHover = (
  color: string,
  isNightMode: boolean,
  disabled: boolean
) => {
  if (disabled) {
    return getColorDisabled(isNightMode);
  }
  switch (color) {
    case 'secondary':
    case 'ghost':
      return isNightMode ? Colors.TEAL_LIGHT : Colors.TEAL;
  }
  return undefined;
};

export const getBackgroundColorHover = (
  color: string,
  isNightMode: boolean,
  disabled: boolean
) => {
  if (disabled) {
    return isNightMode
      ? hexToRGB(Colors.WHITE, 0.12)
      : hexToRGB(Colors.GRAY_900, 0.05);
  }
  switch (color) {
    case 'primary':
      return Colors.TEAL_LIGHT2;
    case 'secondary':
    case 'ghost':
      return isNightMode
        ? hexToRGB(Colors.TEAL_LIGHT, 0.2)
        : hexToRGB(Colors.TEAL, 0.14);
    case 'destructive':
      return Colors.RED_LIGHT;
  }
  return undefined;
};

export const getPadding = (size: string, iconSide: string) => {
  if (iconSide === 'textonly') {
    switch (size) {
      case 'small':
        return '3px 6px';
      case 'medium':
        return '6px 12px';
      case 'large':
        return '10px 16px';
    }
  }
  return '0';
};

export const getPaddingLeft = (size: string, iconSide: string) => {
  if (iconSide === 'textonly') {
    return undefined;
  }
  if (iconSide === 'right') {
    switch (size) {
      case 'small':
        return '6px';
      case 'medium':
        return '12px';
      case 'large':
        return '16px';
    }
  }
  return '0';
};

export const getPaddingRight = (size: string, iconSide: string) => {
  if (iconSide === 'textonly') {
    return undefined;
  }
  if (iconSide === 'left') {
    switch (size) {
      case 'small':
        return '6px';
      case 'medium':
        return '12px';
      case 'large':
        return '16px';
    }
  }
  return '0';
};
