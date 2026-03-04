"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

export default function LegacyPlatformApiPage() {
  const router = useRouter();

  useEffect(() => {
    if (typeof window === "undefined") return;
    const workspace = window.localStorage.getItem(LAST_WORKSPACE_KEY);
    if (!workspace) {
      router.replace("/select-workspace");
      return;
    }
    router.replace(`/w/${workspace}/platform/api-keys`);
  }, [router]);

  return (
    <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
      Redirecting...
    </div>
  );
}

