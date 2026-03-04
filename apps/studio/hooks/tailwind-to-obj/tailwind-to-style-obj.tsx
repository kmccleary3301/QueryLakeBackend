"use client";

type StyleObject = {
  width?: string;
  height?: string;
  padding?: string;
  margin?: string;
  display?: string;
  justifyContent?: string;
	flexDirection?: string;
};

const isNumeric = (str : string) : boolean => {
	const r = parseFloat(str);
  return !isNaN(r) && isFinite(r);
}

const measurementModifier = (modifier: string): string => {
	if (modifier.startsWith('[') && modifier.endsWith(']')) {
		return modifier.slice(1, -1);
	} else if (isNumeric(modifier)) {
		return `${modifier}rem`;
	} else {
		return modifier;
	}
}

const tailwindToStyle = (tailwindString: string): StyleObject => {
  let styleObject: StyleObject = {};
  const classes = tailwindString.split(' ');

  classes.forEach((cls) => {
    const [base, modifier] = cls.split('-');

    switch (base) {
      case 'w':
      case 'h':
        styleObject[base === 'w' ? 'width' : 'height'] = measurementModifier(modifier);
        break;
      case 'p':
      case 'm':
        styleObject[base === 'p' ? 'padding' : 'margin'] = measurementModifier(modifier);
        break;
      case 'flex':
        if (modifier) {
          styleObject.flexDirection = modifier;
        } else {
					styleObject.display = 'flex';
				}
        break;
      case 'justify':
        switch (modifier) {
          case 'between':
            styleObject.justifyContent = 'space-between';
            break;
          case 'end':
            styleObject.justifyContent = 'flex-end';
            break;
          default:
            styleObject.justifyContent = modifier;
            break;
        }
        break;
      default:
        break;
    }
  });

  return styleObject;
};

export default tailwindToStyle;