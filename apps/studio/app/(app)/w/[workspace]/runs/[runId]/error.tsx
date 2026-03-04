"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";

type RunErrorProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function RunError({ error, reset }: RunErrorProps) {
  const params = useParams<{ workspace: string; runId: string }>();
  const workspace = params?.workspace;
  const runId = params?.runId;

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Run page error</h1>
      <p className="text-sm text-muted-foreground">
        Something went wrong while loading this run view.
      </p>
      <pre className="rounded-lg border border-border bg-card/40 p-4 text-xs text-muted-foreground whitespace-pre-wrap">
        {error.message}
      </pre>
      <div className="flex flex-wrap gap-2">
        <Button onClick={reset}>Retry</Button>
        {workspace ? (
          <Button asChild variant="outline">
            <Link href={`/w/${workspace}/runs`}>Back to runs</Link>
          </Button>
        ) : null}
        {workspace && runId ? (
          <Button asChild variant="outline">
            <Link href={`/w/${workspace}/runs/${runId}`}>Reload run</Link>
          </Button>
        ) : null}
        <Button asChild variant="ghost">
          <Link href="/select-workspace">Select workspace</Link>
        </Button>
      </div>
    </div>
  );
}

