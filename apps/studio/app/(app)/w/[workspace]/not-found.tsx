"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";

export default function WorkspaceNotFound() {
  const params = useParams<{ workspace: string }>();
  const workspace = params?.workspace;

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-sm text-muted-foreground">
        The requested workspace page does not exist.
      </p>
      <div className="flex flex-wrap gap-2">
        {workspace ? (
          <Button asChild>
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

