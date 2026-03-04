"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { useContextAction } from "@/app/context-provider";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

export default function WorkspaceIntegrationsPage() {
  const params = useParams<{ workspace: string }>()!;
  const { userData, authReviewed, loginValid } = useContextAction();

  const configuredProviders = useMemo(() => {
    return userData?.user_set_providers ?? [];
  }, [userData?.user_set_providers]);

  if (!authReviewed) {
    return (
      <div className="max-w-4xl space-y-6">
        <div className="h-6 w-48 rounded bg-muted" />
        <div className="h-4 w-72 rounded bg-muted" />
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="max-w-4xl space-y-4">
        <h1 className="text-2xl font-semibold">Integrations</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view integration status.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}/settings`}>Settings</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>Integrations</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Integrations</h1>
        <p className="text-sm text-muted-foreground">
          View integration status and manage external provider keys.
        </p>
      </div>

      {!isPersonalWorkspace(params.workspace) ? (
        <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
          Workspace-level provider keys are not supported yet. User-scoped keys
          apply across all workspaces.
        </div>
      ) : null}

      <div className="rounded-lg border border-border bg-card/40 p-5 space-y-3">
          <div>
            <div className="text-sm font-semibold">Provider keys</div>
            <div className="text-xs text-muted-foreground">
              Configured providers: {configuredProviders.length}
            </div>
          </div>
          {configuredProviders.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {configuredProviders.map((provider) => (
                <span
                  key={provider}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs text-muted-foreground"
                >
                  {provider}
                </span>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              No provider keys configured yet.
            </div>
          )}
          <div className="pt-2">
            <Button asChild size="sm" variant="outline">
              <Link href="/account/providers">Manage provider keys</Link>
            </Button>
          </div>
      </div>
    </div>
  );
}
