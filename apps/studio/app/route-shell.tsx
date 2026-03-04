"use client";

import { usePathname } from "next/navigation";

import SidebarController from "@/components/sidebar/sidebar";

type RouteShellProps = {
  children: React.ReactNode;
};

const NEW_SHELL_PREFIXES = [
  "/w/",
  "/select-workspace",
  "/account",
  "/status",
];

function isNewShell(pathname: string) {
  return NEW_SHELL_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export default function RouteShell({ children }: RouteShellProps) {
  const pathname = usePathname() ?? "";
  if (isNewShell(pathname)) {
    return <>{children}</>;
  }

  return (
    <div className="relative flex h-screen w-screen flex-row bg-background">
      <SidebarController />
      <div className="relative flex h-screen w-full flex-col bg-background text-primary">
        {children}
      </div>
    </div>
  );
}
