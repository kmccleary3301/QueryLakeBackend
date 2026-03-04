export function rgbToHsl(rgb : number[]) {
  let r = rgb[0], g = rgb[1], b = rgb[2];
  r /= 255, g /= 255, b /= 255;
  let max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2 as number;

  if(max == min){
      h = s = 0;  // achromatic
  } else {
      let diff = max - min;
      s = l > 0.5 ? diff / (2 - max - min) : diff / (max + min);
      switch(max){
          case r: h = (g - b) / diff + (g < b ? 6 : 0); break;
          case g: h = (b - r) / diff + 2; break;
          case b: h = (r - g) / diff + 4; break;
      }
      (h as number) /= 6;
  }

  return [Math.round((h as number) * 360 * 10) / 10, Math.round(s * 100 * 10) / 10, Math.round(l * 100 * 10) / 10];
}

function hue2rgb(p : number, q : number, t : number){
  if(t < 0) t += 1;
  if(t > 1) t -= 1;
  if(t < 1/6) return p + ((q - p) * 6 * t);
  if(t < 1/2) return q;
  if(t < 2/3) return p + ((q - p) * ((2 / 3) - t) * 6);
  return p;
};

export function hslToRgb(hsl : number[]) {
  let h = hsl[0], s = hsl[1], l = hsl[2];
  let r, g, b;

  s /= 100;
  l /= 100;

  if(s == 0){
      r = g = b = l; // achromatic
  } else {

      let q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      let p = 2 * l - q;
      r = hue2rgb(p, q, (h / 360) + 1/3);
      g = hue2rgb(p, q, (h / 360));
      b = hue2rgb(p, q, (h / 360) - 1/3);
  }

  return [r * 255, g * 255, b * 255];
}

export function rgbToHex(rgb : number[]) {
  let r = rgb[0], g = rgb[1], b = rgb[2];
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

export function hexToRgb(hex : string) {
  let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
  ] : null;
}

export function hslStringToHsl(hsl : string) {
  let result = /^([\d.]+)\s*([\d.]+)%\s*([\d.]+)%\s*$/i.exec(hsl);
  const final = result ? [
      parseInt(result[1]),
      parseFloat(result[2]),
      parseFloat(result[3])
  ] : null;
  return final;
}

export function hslToHslString(hsl : number[]) {
  return `${hsl[0].toFixed(1)} ${hsl[1].toFixed(1)}% ${hsl[2].toFixed(1)}%`;
}

export function hslStringToRGBHex(hsl : string) {
  let hslValues = hslStringToHsl(hsl);
  if (!hslValues) return null;
  let rgbValues = hslToRgb([hslValues[0], hslValues[1], hslValues[2]]);
  const result = rgbToHex([Math.round(rgbValues[0]), Math.round(rgbValues[1]), Math.round(rgbValues[2])]).toUpperCase();
  return result;
}

export function rbgHexToHslString(hex : string) {
  let rgbValues = hexToRgb(hex);
  if (!rgbValues) return null;
  let hslValues = rgbToHsl([rgbValues[0], rgbValues[1], rgbValues[2]]);
  return hslToHslString(hslValues);
}