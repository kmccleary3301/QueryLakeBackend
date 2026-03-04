// ThemeProvider.tsx
'use client';
import React, {
	Dispatch,
	PropsWithChildren,
	SetStateAction,
	createContext,
	useContext,
	useEffect,
	useState,
} from 'react';
import { themes } from '@/registry/themes';
import { useTheme } from "next-themes"

const REGISTRY_THEMES_PRE = new Map<string, object>();

for (const theme of themes) {
  REGISTRY_THEMES_PRE.set(theme.name, {
    "theme-select-id": theme.name, 
    ...theme.cssVars,
    light: {
      "theme-select-id": theme.name,
      ...theme.cssVars.light
    },
    dark: {
      "theme-select-id": theme.name,
      ...theme.cssVars.dark
    } 
  });
}

export const REGISTRY_THEMES_MAP = REGISTRY_THEMES_PRE;
export const COMBOBOX_THEMES : {label : string, value: string}[] = themes.map((theme) => ({
  label: theme.label,
  value: theme.name
}));
// expport const REGISTRY




export type themeType = {
  "theme-select-id": string,
  "theme-one": string,
  background: string;
  "background-sidebar": string;
  foreground: string;
  card: string;
  "card-foreground": string;
  popover: string;
  "popover-foreground": string;
  primary: string;
  "primary-foreground": string;
  secondary: string;
  "secondary-foreground": string;
  muted: string;
  "muted-foreground": string;
  accent: string;
  "accent-foreground": string;
  destructive: string;
  "destructive-foreground": string;
  border: string;
  input: string;
};

export type dualThemeType = {light: themeType, dark: themeType};

export type registryThemeEntry = {label: string, mode: "light" | "dark", value: string, stylesheet:themeType}

export const REGISTRY_THEMES : registryThemeEntry[] = Array(themes.length*2).fill(0).map((_, i) => {
  const themes_i = themes[Math.floor(i / 2)];
  let result = {...(i % 2 == 0)?themes_i.cssVars.light:themes_i.cssVars.dark, radius: 2}
  if (result.radius) delete (result as {radius: unknown})?.radius;
  
  return {
    label: `${themes_i.label} (${(i % 2 == 0)?"Light":"Dark"})`,
    mode: (i % 2 == 0)?"light":"dark",
    value: themes_i.name,
    stylesheet: {"theme-select-id": themes_i.name, ...result}
  };
}) as unknown as registryThemeEntry[];

const DEFAULT_THEME_ID = "rose";
const DEFAULT_THEME = REGISTRY_THEMES_MAP.get(DEFAULT_THEME_ID) as {light: themeType, dark: themeType};

const Context = createContext<{
	theme: dualThemeType;
  setTheme: Dispatch<SetStateAction<dualThemeType>>;
  themeBrightness: themeType;
  setThemeBrightness: Dispatch<SetStateAction<themeType>>;
  themeStylesheet: React.CSSProperties;
  setThemeStylesheet: Dispatch<SetStateAction<React.CSSProperties>>;
  generateStylesheet: (theme: themeType) => React.CSSProperties;
}>({
	theme: DEFAULT_THEME,
  setTheme: () => {},
  themeBrightness: DEFAULT_THEME.dark,
  setThemeBrightness: () => {},
  themeStylesheet: {},
  setThemeStylesheet: () => {},
  generateStylesheet: () => ({} as React.CSSProperties),
});

export const StateThemeProvider = ({children}: PropsWithChildren<{}>) => {
  const used_theme_next = useTheme();
  const system_mode_theme = used_theme_next.theme,
        system_mode_setTheme = used_theme_next.setTheme;
  
  const generate_stylesheet = (theme: themeType) => {
    return {
      '--theme-one': theme["theme-one"],
      '--background': theme.background,
      '--background-sidebar': theme["background-sidebar"],
      '--foreground': theme.foreground,
      '--card': theme.card,
      '--card-foreground': theme["card-foreground"],
      '--popover': theme.popover,
      '--popover-foreground': theme["popover-foreground"],
      '--primary': theme.primary,
      '--primary-foreground': theme["primary-foreground"],
      '--secondary': theme.secondary,
      '--secondary-foreground': theme["secondary-foreground"],
      '--muted': theme.muted,
      '--muted-foreground': theme["muted-foreground"],
      '--accent': theme.accent,
      '--accent-foreground': theme["accent-foreground"],
      '--destructive': theme.destructive,
      '--destructive-foreground': theme["destructive-foreground"],
      '--border': theme.border,
      '--input': theme.input,
    } as React.CSSProperties;
  }

	const [theme_i, set_theme_i] = useState<dualThemeType>(DEFAULT_THEME);
  const [theme_brightness_i, set_theme_brightness_i] = useState<themeType>(DEFAULT_THEME.dark);
  const [theme_stylesheet_i, set_theme_stylesheet_i] = useState<React.CSSProperties>(generate_stylesheet(DEFAULT_THEME.dark));

  // useEffect(() => {
  //   const dark = (system_mode_theme === "light")?false:true;
  //   const themeGet = REGISTRY_THEMES_MAP.get(DEFAULT_THEME_ID) as {light: themeType, dark: themeType};
  //   set_theme_i(dark?themeGet.dark:themeGet.light);
  // }, []);

  // useEffect(() => {
  //   const dark = (system_mode_theme === "light")?false:true;
  //   set_theme_stylesheet_i(generate_stylesheet(dark?theme_i.dark:theme_i.light));
  // }, [theme_i]);

  useEffect(() => {
    // console.log("SYSTEM THEME:", system_mode_theme);
    const dark = (system_mode_theme === "light")?false:true;
    set_theme_brightness_i(dark?theme_i.dark:theme_i.light);
    set_theme_stylesheet_i(generate_stylesheet(dark?theme_i.dark:theme_i.light));
    // console.log("DARK:", dark);
    // const themeGet = REGISTRY_THEMES_MAP.get(theme_i.dark['theme-select-id']) as {light: themeType, dark: themeType};
    // set_theme_i(themeGet);
  }, [system_mode_theme, theme_i]);

	return (
		<Context.Provider value={{ 
			theme: theme_i,
      setTheme: set_theme_i,
      themeBrightness: theme_brightness_i,
      setThemeBrightness: set_theme_brightness_i,
      themeStylesheet: theme_stylesheet_i,
      setThemeStylesheet: set_theme_stylesheet_i,
      generateStylesheet: generate_stylesheet
		}}>
      <div style={{
      } as React.CSSProperties}>
			  {children}
      </div>
		</Context.Provider>
	);
};

export const useThemeContextAction = () => {
	return useContext(Context);
};

export function ThemeProviderWrapper({children}:{children: React.ReactNode}) {
  const {
    themeStylesheet
  } = useThemeContextAction();

  return (
    <div style={themeStylesheet}>
      {children}
    </div>
  );
}