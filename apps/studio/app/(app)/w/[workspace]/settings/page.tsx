import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

type SettingsPageProps = {
  params: Promise<{ workspace: string }>;
};

export default async function SettingsPage({ params }: SettingsPageProps) {
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
              <BreadcrumbPage>Settings</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Workspace configuration: members, integrations, and access policies.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Members</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Invite teammates, manage roles, and review access.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/settings/members`}>Manage members</Link>
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5">
          <h2 className="text-base font-semibold">Integrations</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Configure provider keys and external service connections.
          </p>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/settings/integrations`}>
                Configure integrations
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

