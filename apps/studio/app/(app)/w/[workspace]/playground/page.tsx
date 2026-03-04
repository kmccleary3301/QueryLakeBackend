"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>Playground</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Playground</h1>
        <p className="text-sm text-muted-foreground">
          Quick access to common workflows and experiments.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">LLM playground</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Use the legacy LLM playground UI (workspace-scoped UI coming next).
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href="/platform/playground/llm">Open legacy playground</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Collections</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload documents, search chunks, and inspect retrieval results.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/collections`}>Browse collections</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Files</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload PDFs and track ingestion status for your workspace.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/files`}>Upload files</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Runs</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Start toolchain runs and inspect sessions and streaming output.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/runs/new`}>Start a run</Link>
            </Button>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
        This page is intentionally lightweight: it links to working surfaces
        (Collections/Files/Runs) and preserves the legacy LLM playground until a
        workspace-scoped replacement is ready.
      </div>
    </div>
  );
}
