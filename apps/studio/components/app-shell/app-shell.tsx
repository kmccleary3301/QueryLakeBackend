"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { useContextAction } from "@/app/context-provider";
import WorkspaceCommandPalette from "@/components/app-shell/workspace-command-palette";

type AppShellProps = {
  children: React.ReactNode;
};

const navItems = [
  { label: "Dashboard", href: "/dashboard" },
  { label: "Collections", href: "/collections" },
  { label: "Files", href: "/files" },
  { label: "Toolchains", href: "/toolchains" },
  { label: "Runs", href: "/runs" },
  { label: "Playground", href: "/playground" },
  { label: "API Keys", href: "/platform/api-keys" },
  { label: "Usage", href: "/platform/usage" },
  { label: "Members", href: "/settings/members" },
  { label: "Integrations", href: "/settings/integrations" },
];

function getWorkspaceFromPath(pathname: string) {
  const segments = pathname.split("/").filter(Boolean);
  if (segments.length >= 2 && segments[0] === "w") {
    return segments[1];
  }
  return "personal";
}

function resolveWorkspaceLabel(workspace: string, userData?: { username: string; memberships: { organization_id: string; organization_name: string }[] }) {
  if (workspace === "personal") {
    return userData?.username ? `${userData.username} (Personal)` : "Personal";
  }
  const match = userData?.memberships?.find(
    (membership) => membership.organization_id === workspace
  );
  return match?.organization_name ?? workspace;
}

export default function AppShell({ children }: AppShellProps) {
  const pathname = usePathname() ?? "";
  const workspace = getWorkspaceFromPath(pathname);
  const { userData } = useContextAction();
  const workspaceLabel = resolveWorkspaceLabel(workspace, userData);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (workspace) {
      window.localStorage.setItem("ql_last_workspace", workspace);
    }
  }, [workspace]);

  return (
    <div className="flex h-screen w-screen bg-background text-foreground">
      <aside className="flex h-full w-64 flex-col border-r border-border px-4 py-4">
        <div className="mb-4">
          <Link href="/select-workspace" className="w-full">
            <Button variant="outline" className="w-full justify-between">
              Workspace: {workspaceLabel}
            </Button>
          </Link>
        </div>
        <nav className="flex flex-col gap-1 text-sm">
          {navItems.map((item) => {
            const href = `/w/${workspace}${item.href}`;
            const active = pathname.startsWith(href);
            return (
              <Link
                key={item.href}
                href={href}
                className={`rounded-md px-3 py-2 transition ${
                  active ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <div className="flex h-full flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-6 py-3">
          <div className="text-sm text-muted-foreground" title={workspace}>
            Workspace: {workspaceLabel}
          </div>
          <div className="flex items-center gap-2">
            <WorkspaceCommandPalette
              workspace={workspace}
              workspaceLabel={workspaceLabel}
            />
            <Button asChild size="sm">
              <Link href={`/w/${workspace}/runs/new`}>New run</Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link href="/account/profile">
                {userData?.username ? userData.username : "Account"}
              </Link>
            </Button>
          </div>
        </header>
        <main className="flex-1 overflow-auto px-6 py-6">{children}</main>
      </div>
    </div>
  );
}
