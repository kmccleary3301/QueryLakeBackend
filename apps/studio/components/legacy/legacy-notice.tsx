"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback } from "react";

import { Button } from "@/components/ui/button";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

type LegacyNoticeProps = {
  title?: string;
  description?: string;
  workspacePath?: `/${string}`;
  ctaLabel?: string;
};

export default function LegacyNotice({
  title = "Legacy page",
  description = "You are viewing a legacy page. The workspace UI is the recommended surface going forward.",
  workspacePath = "/dashboard",
  ctaLabel = "Open in new workspace UI",
}: LegacyNoticeProps) {
  const router = useRouter();

  const openWorkspaceUi = useCallback(() => {
    if (typeof window === "undefined") return;
    const workspace = window.localStorage.getItem(LAST_WORKSPACE_KEY);
    if (!workspace) {
      router.push("/select-workspace");
      return;
    }
    router.push(`/w/${workspace}${workspacePath}`);
  }, [router, workspacePath]);

  return (
    <div className="rounded-lg border border-dashed border-border bg-card/40 p-4 text-sm text-muted-foreground">
      <div className="font-medium text-foreground">{title}</div>
      <p className="mt-1">{description}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" variant="outline" onClick={openWorkspaceUi}>
          {ctaLabel}
        </Button>
        <Button asChild size="sm" variant="ghost">
          <Link href="/select-workspace">Select workspace</Link>
        </Button>
      </div>
    </div>
  );
}

