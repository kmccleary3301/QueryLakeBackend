import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

type PlatformPageProps = {
  params: Promise<{ workspace: string }>;
};

export default async function PlatformPage({ params }: PlatformPageProps) {
  const { workspace } = await params;

  return (
    <div className="space-y-6">
      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${workspace}`}>Workspace</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>Platform</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Platform</h1>
        <p className="text-sm text-muted-foreground">
          Account-level platform tools for API access and usage visibility.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">API keys</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Create and revoke API keys used for programmatic access.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/platform/api-keys`}>Manage API keys</Link>
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Usage</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            View and export usage metrics for this workspace.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/platform/usage`}>View usage</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

