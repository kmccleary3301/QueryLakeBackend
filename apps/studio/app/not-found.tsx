"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback } from "react";

import { Button } from "@/components/ui/button";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

export default function NotFound() {
  const router = useRouter();

  const openLastWorkspace = useCallback(() => {
    if (typeof window === "undefined") return;
    const workspace = window.localStorage.getItem(LAST_WORKSPACE_KEY);
    if (!workspace) {
      router.push("/select-workspace");
      return;
    }
    router.push(`/w/${workspace}/dashboard`);
  }, [router]);

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-sm text-muted-foreground">
        The page you requested does not exist.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button onClick={openLastWorkspace}>Go to last workspace</Button>
        <Button asChild variant="outline">
          <Link href="/select-workspace">Select workspace</Link>
        </Button>
      </div>
    </div>
  );
}

