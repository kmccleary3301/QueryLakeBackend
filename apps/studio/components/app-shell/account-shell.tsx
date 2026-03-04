"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type AccountShellProps = {
  children: React.ReactNode;
};

const navItems = [
  { label: "Profile", href: "/account/profile" },
  { label: "Security", href: "/account/security" },
  { label: "Preferences", href: "/account/preferences" },
  { label: "Providers", href: "/account/providers" },
];

export default function AccountShell({ children }: AccountShellProps) {
  const pathname = usePathname() ?? "";

  return (
    <div className="flex h-screen w-screen bg-background text-foreground">
      <aside className="flex h-full w-64 flex-col border-r border-border px-4 py-4">
        <div className="mb-4 text-sm font-medium text-foreground">Account</div>
        <nav className="flex flex-col gap-1 text-sm">
          {navItems.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-md px-3 py-2 transition ${
                  active
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted/50"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto text-xs text-muted-foreground">
          Back to workspace: <Link href="/select-workspace" className="underline">switch</Link>
        </div>
      </aside>
      <main className="flex-1 overflow-auto px-6 py-6">{children}</main>
    </div>
  );
}
