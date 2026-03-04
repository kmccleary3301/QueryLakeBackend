"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";

type WorkspaceErrorProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function WorkspaceError({ error, reset }: WorkspaceErrorProps) {
  const params = useParams<{ workspace: string }>();
  const workspace = params?.workspace;

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Something went wrong</h1>
      <p className="text-sm text-muted-foreground">
        An unexpected error occurred while loading this workspace page.
      </p>
      <pre className="rounded-lg border border-border bg-card/40 p-4 text-xs text-muted-foreground whitespace-pre-wrap">
        {error.message}
      </pre>
      <div className="flex flex-wrap gap-2">
        <Button onClick={reset}>Retry</Button>
        {workspace ? (
          <Button asChild variant="outline">
            <Link href={`/w/${workspace}/dashboard`}>Back to dashboard</Link>
          </Button>
        ) : null}
        <Button asChild variant="outline">
          <Link href="/select-workspace">Select workspace</Link>
        </Button>
      </div>
    </div>
  );
}

