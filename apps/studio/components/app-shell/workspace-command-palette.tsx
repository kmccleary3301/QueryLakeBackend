"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { cn } from "@/lib/utils";

type WorkspaceCommandPaletteProps = {
  workspace: string;
  workspaceLabel: string;
};

type CommandEntry = {
  label: string;
  keywords?: string[];
  onSelect: () => void;
  shortcut?: string;
};

export default function WorkspaceCommandPalette({
  workspace,
  workspaceLabel,
}: WorkspaceCommandPaletteProps) {
  const router = useRouter();
  const { setTheme } = useTheme();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!(event.key === "k" && (event.metaKey || event.ctrlKey))) return;
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target instanceof HTMLInputElement ||
          target instanceof HTMLTextAreaElement ||
          target instanceof HTMLSelectElement)
      ) {
        return;
      }
      event.preventDefault();
      setOpen((prev) => !prev);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  const runCommand = useCallback((command: () => void) => {
    setOpen(false);
    command();
  }, []);

  const workspaceCommands = useMemo<CommandEntry[]>(() => {
    const base = `/w/${workspace}`;
    return [
      { label: "Dashboard", onSelect: () => router.push(`${base}/dashboard`) },
      { label: "Collections", onSelect: () => router.push(`${base}/collections`) },
      { label: "Create collection", onSelect: () => router.push(`${base}/collections/new`), shortcut: "N C" },
      { label: "Files", onSelect: () => router.push(`${base}/files`) },
      { label: "Toolchains", onSelect: () => router.push(`${base}/toolchains`) },
      { label: "Runs", onSelect: () => router.push(`${base}/runs`) },
      { label: "New run", onSelect: () => router.push(`${base}/runs/new`), shortcut: "N R" },
      { label: "Platform", onSelect: () => router.push(`${base}/platform`) },
      { label: "API keys", onSelect: () => router.push(`${base}/platform/api-keys`) },
      { label: "Usage", onSelect: () => router.push(`${base}/platform/usage`) },
      { label: "Settings", onSelect: () => router.push(`${base}/settings`) },
      { label: "Members", onSelect: () => router.push(`${base}/settings/members`) },
      { label: "Integrations", onSelect: () => router.push(`${base}/settings/integrations`) },
      { label: "Playground", onSelect: () => router.push(`${base}/playground`) },
    ];
  }, [router, workspace]);

  const globalCommands = useMemo<CommandEntry[]>(
    () => [
      { label: "Select workspace", onSelect: () => router.push("/select-workspace") },
      { label: "System status", onSelect: () => router.push("/status") },
      { label: "Docs", onSelect: () => router.push("/docs") },
    ],
    [router]
  );

  const accountCommands = useMemo<CommandEntry[]>(
    () => [
      { label: "Account profile", onSelect: () => router.push("/account/profile") },
      { label: "Account security", onSelect: () => router.push("/account/security") },
      { label: "Account preferences", onSelect: () => router.push("/account/preferences") },
      { label: "Account providers", onSelect: () => router.push("/account/providers") },
    ],
    [router]
  );

  const themeCommands = useMemo<CommandEntry[]>(
    () => [
      { label: "Theme: System", onSelect: () => setTheme("system") },
      { label: "Theme: Light", onSelect: () => setTheme("light") },
      { label: "Theme: Dark", onSelect: () => setTheme("dark") },
    ],
    [setTheme]
  );

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        className={cn("relative")}
        onClick={() => setOpen(true)}
        title="Command palette (Ctrl/Cmd+K)"
      >
        Command
        <kbd className="pointer-events-none ml-2 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground sm:inline-flex">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>

      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Type a command or search..." />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>

          <CommandGroup heading={`Workspace (${workspaceLabel})`}>
            {workspaceCommands.map((entry) => (
              <CommandItem
                key={entry.label}
                value={[entry.label, ...(entry.keywords ?? [])].join(" ")}
                onSelect={() => runCommand(entry.onSelect)}
              >
                {entry.label}
                {entry.shortcut ? (
                  <CommandShortcut>{entry.shortcut}</CommandShortcut>
                ) : null}
              </CommandItem>
            ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Account">
            {accountCommands.map((entry) => (
              <CommandItem
                key={entry.label}
                value={entry.label}
                onSelect={() => runCommand(entry.onSelect)}
              >
                {entry.label}
              </CommandItem>
            ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Global">
            {globalCommands.map((entry) => (
              <CommandItem
                key={entry.label}
                value={entry.label}
                onSelect={() => runCommand(entry.onSelect)}
              >
                {entry.label}
              </CommandItem>
            ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Theme">
            {themeCommands.map((entry) => (
              <CommandItem
                key={entry.label}
                value={entry.label}
                onSelect={() => runCommand(entry.onSelect)}
              >
                {entry.label}
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  );
}

