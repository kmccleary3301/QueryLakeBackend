"use client";

import Link from "next/link";

import { Button } from "@/components/ui/button";
import { useContextAction } from "@/app/context-provider";

export default function AccountProfilePage() {
  const { userData, authReviewed, loginValid } = useContextAction();

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
        <h1 className="text-2xl font-semibold">Profile</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view your account profile.
        </p>
        <Button asChild size="sm">
          <Link href="/auth/login">Go to login</Link>
        </Button>
      </div>
    );
  }

  const membershipCount = userData.memberships?.length ?? 0;
  const providerCount = userData.providers?.length ?? 0;
  const configuredProviderCount = userData.user_set_providers?.length ?? 0;

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Profile</h1>
        <p className="text-sm text-muted-foreground">
          View your account identity and configuration summary.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5">
          <div className="text-sm font-medium text-muted-foreground">
            Username
          </div>
          <div className="mt-1 text-base font-semibold">{userData.username}</div>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5">
          <div className="text-sm font-medium text-muted-foreground">
            Workspaces
          </div>
          <div className="mt-1 text-base font-semibold">
            {membershipCount + 1}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            Personal workspace + {membershipCount} organization membership(s)
          </div>
          <div className="mt-3">
            <Button asChild size="sm" variant="outline">
              <Link href="/select-workspace">Select workspace</Link>
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5">
          <div className="text-sm font-medium text-muted-foreground">
            External providers
          </div>
          <div className="mt-1 text-base font-semibold">{providerCount}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {configuredProviderCount} configured with keys
          </div>
          <div className="mt-3">
            <Button asChild size="sm" variant="outline">
              <Link href="/account/providers">Manage provider keys</Link>
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5">
          <div className="text-sm font-medium text-muted-foreground">
            Default toolchain
          </div>
          <div className="mt-1 text-base font-semibold">
            {userData.default_toolchain?.title ?? "—"}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            Used to preselect toolchains for new runs.
          </div>
        </div>
      </div>
    </div>
  );
}
