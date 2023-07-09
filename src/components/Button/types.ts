export const ButtonSizes = {
  Small: 'small',
  Medium: 'medium',
  Large: 'large',
} as const;
export type ButtonSize = (typeof ButtonSizes)[keyof typeof ButtonSizes];

export const ButtonIconSides = {
  Left: 'left',
  Right: 'right',
} as const;
export type ButtonIconSide =
  (typeof ButtonIconSides)[keyof typeof ButtonIconSides];

export const ButtonVariants = {
  Primary: 'primary',
  Secondary: 'secondary',
  Ghost: 'ghost',
  Destructive: 'destructive',
} as const;
export type ButtonVariant =
  (typeof ButtonVariants)[keyof typeof ButtonVariants];
