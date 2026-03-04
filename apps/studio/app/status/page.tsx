"use client";

import Link from "next/link";
import { useCallback, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { useContextAction } from "@/app/context-provider";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

export default function StatusPage() {
  const { authReviewed, loginValid, userData } = useContextAction();
  const [copied, setCopied] = useState(false);

  const diagnosticsText = useMemo(() => {
    const now = new Date().toISOString();
    const hasWindow = typeof window !== "undefined";
    const lastWorkspace = hasWindow
      ? window.localStorage.getItem(LAST_WORKSPACE_KEY)
      : null;
    const location = hasWindow ? window.location.href : "(server)";
    const userAgent = hasWindow ? window.navigator.userAgent : "(server)";

    return [
      `QueryLake Frontend Diagnostics`,
      `timestamp=${now}`,
      `location=${location}`,
      `userAgent=${userAgent}`,
      `authReviewed=${authReviewed}`,
      `loginValid=${loginValid}`,
      `username=${userData?.username ?? "(none)"}`,
      `lastWorkspace=${lastWorkspace ?? "(none)"}`,
      ``,
      `Backend links (open in browser):`,
      `healthz=http://localhost:8000/healthz`,
      `readyz=http://localhost:8000/readyz`,
    ].join("\n");
  }, [authReviewed, loginValid, userData?.username]);

  const copyDiagnostics = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(diagnosticsText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = diagnosticsText;
      textarea.style.position = "fixed";
      textarea.style.top = "0";
      textarea.style.left = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }, [diagnosticsText]);

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">System status</h1>
        <p className="text-sm text-muted-foreground">
          Health and connectivity checks for the QueryLake backend.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Backend health</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Validate that the API gateway and model services are reachable.
          </p>
          <div className="mt-4 flex gap-2">
            <Button asChild size="sm" variant="outline">
              <Link href="http://localhost:8000/healthz">Healthz</Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link href="http://localhost:8000/readyz">Readyz</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Diagnostics</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Copy a diagnostic snapshot for debugging.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={copyDiagnostics}>
              {copied ? "Copied" : "Copy diagnostics"}
            </Button>
            <Button asChild size="sm" variant="ghost">
              <Link href="/select-workspace">Select workspace</Link>
            </Button>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground whitespace-pre-wrap">
        {diagnosticsText}
      </div>
    </div>
  );
}
