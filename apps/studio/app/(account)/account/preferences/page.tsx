"use client";

import { useMemo } from "react";
import { useTheme } from "next-themes";
import type { BundledTheme } from "shiki/themes";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import MarkdownCodeBlock from "@/components/markdown/markdown-code-block";
import { useContextAction } from "@/app/context-provider";
import {
  COMBOBOX_THEMES,
  REGISTRY_THEMES_MAP,
  useThemeContextAction,
} from "@/app/theme-provider";
import type { dualThemeType } from "@/app/theme-provider";
import {
  SHIKI_THEMES,
  SHIKI_THEMES_BACKGROUND_COLORS,
  SHIKI_THEMES_TEXT_COLORS,
} from "@/lib/shiki";

const DEMO_CODE = `function hello(name) {
  return "Hello, " + name + "!";
}

console.log(hello("QueryLake"));`;

export default function AccountPreferencesPage() {
  const { authReviewed, loginValid, userData, shikiTheme, setShikiTheme } =
    useContextAction();
  const { theme: modeTheme, setTheme: setModeTheme } = useTheme();
  const { theme, setTheme } = useThemeContextAction();

  const currentRegistryTheme = theme?.dark?.["theme-select-id"] ?? "rose";
  const codeThemeValue = (shikiTheme?.theme ?? "tokyo-night") as BundledTheme;

  const providerKeysConfigured = useMemo(() => {
    if (!userData) return 0;
    return userData.user_set_providers?.length ?? 0;
  }, [userData]);

  if (!authReviewed) {
    return (
      <div className="max-w-4xl space-y-6">
        <div className="h-6 w-48 rounded bg-muted" />
        <div className="h-4 w-72 rounded bg-muted" />
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="max-w-4xl space-y-4">
        <h1 className="text-2xl font-semibold">Preferences</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to manage your preferences.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Preferences</h1>
        <p className="text-sm text-muted-foreground">
          Customize appearance and editor defaults for your account.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5 space-y-3">
          <div>
            <div className="text-sm font-semibold">Color mode</div>
            <div className="text-xs text-muted-foreground">
              Controls light/dark/system rendering.
            </div>
          </div>
          <Select
            value={modeTheme ?? "system"}
            onValueChange={(value) => setModeTheme(value)}
          >
            <SelectTrigger className="w-full max-w-[240px]">
              <SelectValue placeholder="Select mode" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="system">System</SelectItem>
              <SelectItem value="light">Light</SelectItem>
              <SelectItem value="dark">Dark</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5 space-y-3">
          <div>
            <div className="text-sm font-semibold">UI theme</div>
            <div className="text-xs text-muted-foreground">
              Applies QueryLake CSS variables across the app.
            </div>
          </div>
          <Select
            value={currentRegistryTheme}
            onValueChange={(value) => {
              const themeGet = REGISTRY_THEMES_MAP.get(value) as
                | dualThemeType
                | undefined;
              if (!themeGet) return;
              setTheme(themeGet);
            }}
          >
            <SelectTrigger className="w-full max-w-[280px]">
              <SelectValue placeholder="Select theme" />
            </SelectTrigger>
            <SelectContent>
              {COMBOBOX_THEMES.map((entry) => (
                <SelectItem key={entry.value} value={entry.value}>
                  {entry.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5 space-y-3 md:col-span-2">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">Code highlight theme</div>
              <div className="text-xs text-muted-foreground">
                Used for markdown code blocks and docs rendering.
              </div>
            </div>
            <Select
              value={codeThemeValue}
              onValueChange={(value) => {
                setShikiTheme({
                  theme: value as BundledTheme,
                  backgroundColor: SHIKI_THEMES_BACKGROUND_COLORS.get(
                    value as BundledTheme
                  ),
                  textColor: SHIKI_THEMES_TEXT_COLORS.get(value as BundledTheme),
                });
              }}
            >
              <SelectTrigger className="w-[280px]">
                <SelectValue placeholder="Select code theme" />
              </SelectTrigger>
              <SelectContent>
                {SHIKI_THEMES.map((entry) => (
                  <SelectItem key={entry.value} value={entry.value}>
                    {entry.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <MarkdownCodeBlock
            text={DEMO_CODE}
            lang="javascript"
            className="max-w-[560px]"
          />
        </div>

        <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground md:col-span-2">
          Provider keys configured: {providerKeysConfigured}. Manage them in{" "}
          <span className="text-foreground">Account → Providers</span>.
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setModeTheme("system");
            const defaultTheme = REGISTRY_THEMES_MAP.get("rose") as
              | dualThemeType
              | undefined;
            if (defaultTheme) setTheme(defaultTheme);
            setShikiTheme({
              theme: "tokyo-night" as BundledTheme,
              backgroundColor: SHIKI_THEMES_BACKGROUND_COLORS.get("tokyo-night"),
              textColor: SHIKI_THEMES_TEXT_COLORS.get("tokyo-night"),
            });
          }}
        >
          Reset to defaults
        </Button>
      </div>
    </div>
  );
}
