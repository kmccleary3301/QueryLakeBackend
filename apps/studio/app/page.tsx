"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

export default function IndexPage() {
  const router = useRouter();

  useEffect(() => {
    if (typeof window === "undefined") return;
    const lastWorkspace = window.localStorage.getItem(LAST_WORKSPACE_KEY);
    if (lastWorkspace) {
      router.replace(`/w/${lastWorkspace}/dashboard`);
    } else {
      router.replace("/select-workspace");
    }
  }, [router]);

  return (
    <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
      Redirecting...
    </div>
  );
}
