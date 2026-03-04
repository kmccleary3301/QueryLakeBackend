import type { BundledLanguage, BundledTheme } from 'shiki'
import { codeToHtml } from 'shiki'

const languages = [
  ["ABAP", "abap"],
  ["ActionScript", "actionscript-3"],
  ["Ada", "ada"],
  ["Angular HTML", "angular-html"],
  ["Angular TypeScript", "angular-ts"],
  ["Apache Conf", "apache"],
  ["Apex", "apex"],
  ["APL", "apl"],
  ["AppleScript", "applescript"],
  ["Ara", "ara"],
  ["Assembly", "asm", "nasm"],
  ["Astro", "astro"],
  ["AWK", "awk"],
  ["Ballerina", "ballerina"],
  ["Batch File", "bat", "batch"],
  ["Beancount", "beancount"],
  ["Berry", "berry", "be"],
  ["BibTeX", "bibtex"],
  ["Bicep", "bicep"],
  ["Blade", "blade"],
  ["C", "c"],
  ["Cadence", "cadence", "cdc"],
  ["Clarity", "clarity"],
  ["Clojure", "clojure", "clj"],
  ["CMake", "cmake"],
  ["COBOL", "cobol"],
  ["CodeQL", "codeql", "ql"],
  ["CoffeeScript", "coffee", "coffeescript"],
  ["C++", "cpp", "c++"],
  ["Crystal", "crystal"],
  ["C#", "csharp", "c#cs"],
  ["CSS", "css"],
  ["CSV", "csv"],
  ["CUE", "cue"],
  ["Cypher", "cypher", "cql"],
  ["D", "d"],
  ["Dart", "dart"],
  ["DAX", "dax"],
  ["Diff", "diff"],
  ["Dockerfile", "docker", "dockerfile"],
  ["Dream Maker", "dream-maker"],
  ["Elixir", "elixir"],
  ["Elm", "elm"],
  ["ERB", "erb"],
  ["Erlang", "erlang", "erl"],
  ["Fish", "fish"],
  ["Fortran (Fixed Form)", "fortran-fixed-form", "fforf77"],
  ["Fortran (Free Form)", "fortran-free-form", "f90f95f03f08f18"],
  ["F#", "fsharp", "f#fs"],
  ["GDResource", "gdresource"],
  ["GDScript", "gdscript"],
  ["GDShader", "gdshader"],
  ["Gherkin", "gherkin"],
  ["Git Commit Message", "git-commit"],
  ["Git Rebase Message", "git-rebase"],
  ["Gleam", "gleam"],
  ["Glimmer JS", "glimmer-js", "gjs"],
  ["Glimmer TS", "glimmer-ts", "gts"],
  ["GLSL", "glsl"],
  ["Gnuplot", "gnuplot"],
  ["Go", "go"],
  ["GraphQL", "graphql", "gql"],
  ["Groovy", "groovy"],
  ["Hack", "hack"],
  ["Ruby Haml", "haml"],
  ["Handlebars", "handlebars", "hbs"],
  ["Haskell", "haskell", "hs"],
  ["HashiCorp HCL", "hcl"],
  ["Hjson", "hjson"],
  ["HLSL", "hlsl"],
  ["HTML", "html"],
  ["HTML (Derivative)", "html-derivative"],
  ["HTTP", "http"],
  ["Imba", "imba"],
  ["INI", "ini", "properties"],
  ["Java", "java"],
  ["JavaScript", "javascript", "js"],
  ["Jinja", "jinja"],
  ["Jison", "jison"],
  ["JSON", "json"],
  ["JSON5", "json5"],
  ["JSON with Comments", "jsonc"],
  ["JSON Lines", "jsonl"],
  ["Jsonnet", "jsonnet"],
  ["JSSM", "jssm", "fsl"],
  ["JSX", "jsx"],
  ["Julia", "julia", "jl"],
  ["Kotlin", "kotlin", "kt", "kts"],
  ["Kusto", "kusto", "kql"],
  ["LaTeX", "latex"],
  ["Less", "less"],
  ["Liquid", "liquid"],
  ["Lisp", "lisp"],
  ["Logo", "logo"],
  ["Lua", "lua"],
  ["Makefile", "make", "makefile"],
  ["Markdown", "markdown", "md"],
  ["Marko", "marko"],
  ["MATLAB", "matlab"],
  ["MDC", "mdc"],
  ["MDX", "mdx"],
  ["Mermaid", "mermaid"],
  ["Mojo", "mojo"],
  ["Move", "move"],
  ["Narrat Language", "narrat", "nar"],
  ["Nextflow", "nextflow", "nf"],
  ["Nginx", "nginx"],
  ["Nim", "nim"],
  ["Nix", "nix"],
  ["Nushell", "nushell", "nu"],
  ["Objective-C", "objective-c", "objc"],
  ["Objective-C++", "objective-cpp"],
  ["OCaml", "ocaml"],
  ["Pascal", "pascal"],
  ["Perl", "perl"],
  ["PHP", "php"],
  ["PL/SQL", "plsql"],
  ["PostCSS", "postcss"],
  ["PowerQuery", "powerquery"],
  ["PowerShell", "powershell", "ps", "ps1"],
  ["Prisma", "prisma"],
  ["Prolog", "prolog"],
  ["Protocol Buffer 3", "proto"],
  ["Pug", "pug", "jade"],
  ["Puppet", "puppet"],
  ["PureScript", "purescript"],
  ["Python", "python", "py"],
  ["R", "r"],
  ["Raku", "raku", "perl6"],
  ["ASP.NET Razor", "razor"],
  ["Windows Registry Script", "reg"],
  ["Rel", "rel"],
  ["RISC-V", "riscv"],
  ["reStructuredText", "rst"],
  ["Ruby", "ruby", "rb"],
  ["Rust", "rust", "rs"],
  ["SAS", "sas"],
  ["Sass", "sass"],
  ["Scala", "scala"],
  ["Scheme", "scheme"],
  ["SCSS", "scss"],
  ["ShaderLab", "shaderlab", "shader"],
  ["Shell", "shellscript", "bash", "sh", "zsh"],
  ["Shell Session", "shellsession", "console"],
  ["Smalltalk", "smalltalk"],
  ["Solidity", "solidity"],
  ["SPARQL", "sparql"],
  ["Splunk Query Language", "splunk", "spl"],
  ["SQL", "sql"],
  ["SSH Config", "ssh-config"],
  ["Stata", "stata"],
  ["Stylus", "stylus", "st"],
  ["Svelte", "svelte"],
  ["Swift", "swift"],
  ["SystemVerilog", "system-verilog"],
  ["Tasl", "tasl"],
  ["Tcl", "tcl"],
  ["Terraform", "terraform", "tft", "tfvars"],
  ["TeX", "tex"],
  ["TOML", "toml"],
  ["TSV", "tsv"],
  ["TSX", "tsx"],
  ["Turtle", "turtle"],
  ["Twig", "twig"],
  ["TypeScript", "typescript", "ts"],
  ["Typst", "typst", "typ"],
  ["V", "v"],
  ["Visual Basic", "vb", "cmd"],
  ["Verilog", "verilog"],
  ["VHDL", "vhdl"],
  ["Vim Script", "viml", "vim", "vimscript"],
  ["Vue", "vue"],
  ["Vue HTML", "vue-html"],
  ["Vyper", "vyper", "vy"],
  ["WebAssembly", "wasm"],
  ["Wenyan", "wenyan"],
  ["WGSL", "wgsl"],
  ["Wolfram", "wolfram", "wl"],
  ["XML", "xml"],
  ["XSL", "xsl"],
  ["YAML", "yaml", "yml"],
  ["ZenScript", "zenscript"],
  ["Zig", "zig"]
];

const SHIKI_BUNDLED_THEMES : {value: BundledTheme, backgroundColor: string, textColor: string}[] = [
  { "value": "andromeeda", "backgroundColor": "#23262E", "textColor": "#D5CED9" },
  { "value": "aurora-x", "backgroundColor": "#07090F", "textColor": "#bbbbbb" },
  { "value": "ayu-dark", "backgroundColor": "#0b0e14", "textColor": "#bfbdb6" },
  { "value": "catppuccin-frappe", "backgroundColor": "#303446", "textColor": "#c6d0f5" },
  { "value": "catppuccin-latte", "backgroundColor": "#eff1f5", "textColor": "#4c4f69" },
  { "value": "catppuccin-macchiato", "backgroundColor": "#24273a", "textColor": "#cad3f5" },
  { "value": "catppuccin-mocha", "backgroundColor": "#1e1e2e", "textColor": "#cdd6f4" },
  { "value": "dark-plus", "backgroundColor": "#1E1E1E", "textColor": "#D4D4D4" },
  { "value": "dracula", "backgroundColor": "#282A36", "textColor": "#F8F8F2" },
  { "value": "dracula-soft", "backgroundColor": "#282A36", "textColor": "#f6f6f4" },
  { "value": "github-dark", "backgroundColor": "#24292e", "textColor": "#e1e4e8" },
  { "value": "github-dark-default", "backgroundColor": "#0d1117", "textColor": "#e6edf3" },
  { "value": "github-dark-dimmed", "backgroundColor": "#22272e", "textColor": "#adbac7" },
  { "value": "github-light", "backgroundColor": "#fff", "textColor": "#24292e" },
  { "value": "github-light-default", "backgroundColor": "#ffffff", "textColor": "#1f2328" },
  { "value": "houston", "backgroundColor": "#17191e", "textColor": "#eef0f9" },
  { "value": "light-plus", "backgroundColor": "#FFFFFF", "textColor": "#000000" },
  { "value": "material-theme", "backgroundColor": "#263238", "textColor": "#EEFFFF" },
  { "value": "material-theme-darker", "backgroundColor": "#212121", "textColor": "#EEFFFF" },
  { "value": "material-theme-lighter", "backgroundColor": "#FAFAFA", "textColor": "#90A4AE" },
  { "value": "material-theme-ocean", "backgroundColor": "#0F111A", "textColor": "#babed8" },
  { "value": "material-theme-palenight", "backgroundColor": "#292D3E", "textColor": "#babed8" },
  { "value": "min-dark", "backgroundColor": "#1f1f1f", "textColor": "#b392f0" },
  { "value": "min-light", "backgroundColor": "#ffffff", "textColor": "#24292eff" },
  { "value": "monokai", "backgroundColor": "#272822", "textColor": "#F8F8F2" },
  { "value": "night-owl", "backgroundColor": "#011627", "textColor": "#d6deeb" },
  { "value": "nord", "backgroundColor": "#2e3440ff", "textColor": "#d8dee9ff" },
  { "value": "one-dark-pro", "backgroundColor": "#282c34", "textColor": "#abb2bf" },
  { "value": "poimandres", "backgroundColor": "#1b1e28", "textColor": "#a6accd" },
  { "value": "red", "backgroundColor": "#390000", "textColor": "#F8F8F8" },
  { "value": "rose-pine", "backgroundColor": "#191724", "textColor": "#e0def4" },
  { "value": "rose-pine-dawn", "backgroundColor": "#faf4ed", "textColor": "#575279" },
  { "value": "rose-pine-moon", "backgroundColor": "#232136", "textColor": "#e0def4" },
  { "value": "slack-dark", "backgroundColor": "#222222", "textColor": "#E6E6E6" },
  { "value": "slack-ochin", "backgroundColor": "#FFF", "textColor": "#002339" },
  { "value": "solarized-dark", "backgroundColor": "#002B36", "textColor": "#839496" },
  { "value": "solarized-light", "backgroundColor": "#FDF6E3", "textColor": "#657B83" },
  { "value": "synthwave-84", "backgroundColor": "#262335", "textColor": "#bbbbbb" },
  { "value": "tokyo-night", "backgroundColor": "#1a1b26", "textColor": "#a9b1d6" },
  { "value": "vesper", "backgroundColor": "#101010", "textColor": "#FFF" },
  { "value": "vitesse-black", "backgroundColor": "#000", "textColor": "#dbd7cacc" },
  { "value": "vitesse-dark", "backgroundColor": "#121212", "textColor": "#dbd7caee" },
  { "value": "vitesse-light", "backgroundColor": "#ffffff", "textColor": "#393a34" }
];

export const SHIKI_THEMES : {value: BundledTheme, label: string}[] = SHIKI_BUNDLED_THEMES.map(theme => ({
  value: theme.value as BundledTheme,
  label: theme.value.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
}));

let SHIKI_THEMES_BACKGROUND_COLORS_PRE = new Map<BundledTheme, string>(),
    SHIKI_THEMES_TEXT_COLORS_PRE = new Map<BundledTheme, string>();
for (const theme of SHIKI_BUNDLED_THEMES) {
  SHIKI_THEMES_BACKGROUND_COLORS_PRE.set(theme.value as BundledTheme, theme.backgroundColor);
  SHIKI_THEMES_TEXT_COLORS_PRE.set(theme.value as BundledTheme, theme.textColor);
}
export const SHIKI_THEMES_BACKGROUND_COLORS = SHIKI_THEMES_BACKGROUND_COLORS_PRE;
export const SHIKI_THEMES_TEXT_COLORS = SHIKI_THEMES_TEXT_COLORS_PRE;

let LANGUAGES_MAP_PRE = new Map<string, {value: BundledLanguage, preview: string}>();

// Loop through each language pair
for (let value_set of languages) {
  // Strip non-alphanumeric characters and force to lowercase
	for (let value of value_set) {
		// const cleanKey = value.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    const cleanKey = value.toLowerCase();
		// Add the cleaned key-value pair to the map

		for (let i = 1; i < cleanKey.length-1; i++) {
			if (LANGUAGES_MAP_PRE.has(cleanKey.slice(0, i))) {
				continue;
			}
			LANGUAGES_MAP_PRE.set(cleanKey.slice(0, i), {value: value_set[1] as BundledLanguage, preview: value_set[0]});
		}
		LANGUAGES_MAP_PRE.set(cleanKey, {value: value_set[1] as BundledLanguage, preview: value_set[0]});
	}
}

const LANGUAGES_MAP = LANGUAGES_MAP_PRE;

export function getLanguage(lang: string) : {value: BundledLanguage | "text", preview: string} {
	const langSimple = (lang || "").replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
	// console.log(LANGUAGES_MAP);
	// console.log("Searching for", langSimple, "got:", LANGUAGES_MAP.get(langSimple));
	return LANGUAGES_MAP.get(langSimple) || {value: "text", preview: "Text"};
}

export async function highlight(code: string, theme: BundledTheme, lang: string) {
	const html = await codeToHtml(code, {
		lang: lang as BundledLanguage,
		theme: theme,
		// colorReplacements: {
		// 	"#1a1b26": "#00000000"
		// },
	});
  return html.replace(/^<pre[^>]*>/i, '<pre>');
}

const test_code = `
print("Hello, World!")
`

export async function get_all_language_backgrounds() {
  const backgrounds = new Map<string, string>();
  let background_entries = new Array<{value: BundledTheme, backgroundColor: string, textColor: string}>();
  
  for (const e of SHIKI_BUNDLED_THEMES) {
    const html = await codeToHtml("test", {
      lang: "python",
      theme: e.value,
      // colorReplacements: {
      //   "#1a1b26": "#00000000"
      // },
    })
    const background = html.match(/^<pre[^>]*>/i);
    console.log("Shiki background:", background);

    const backgroundColor = background?background[0].match(/background\-color\:\#[a-fA-F0-9]+/i):undefined;
    const textColor = background?background[0].match(/\;color\:\#[a-fA-F0-9]+/i):undefined;
    if (backgroundColor && textColor) {
      const b_color_entry = backgroundColor[0].split(":")[1];
      const color_entry = textColor[0].split(":")[1];
      // console.log(color_entry);
      // backgrounds.set(e.value, background[1]);
      background_entries.push({value: e.value, backgroundColor: b_color_entry, textColor: color_entry});
    }
  }
  // const languageBackgrounds = Object.fromEntries(backgrounds) as { [key: BundledLanguage]: string };
  // return languageBackgrounds;
  console.log("Shiki background values:", background_entries);

  return backgrounds;
}