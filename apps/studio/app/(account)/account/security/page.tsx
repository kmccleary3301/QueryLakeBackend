"use client";

import Link from "next/link";

import { Button } from "@/components/ui/button";
import { useContextAction } from "@/app/context-provider";

export default function AccountSecurityPage() {
  const { authReviewed, loginValid, userData } = useContextAction();

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
        <h1 className="text-2xl font-semibold">Security</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view security guidance and account posture.
        </p>
        <Button asChild size="sm">
          <Link href="/auth/login">Go to login</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Security</h1>
        <p className="text-sm text-muted-foreground">
          Recommended practices for protecting your account and API access.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-5 space-y-2">
          <div className="text-sm font-semibold">Provider keys</div>
          <p className="text-sm text-muted-foreground">
            Keep external provider keys private and rotate them if you suspect
            compromise.
          </p>
          <Button asChild size="sm" variant="outline">
            <Link href="/account/providers">Manage provider keys</Link>
          </Button>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-5 space-y-2">
          <div className="text-sm font-semibold">API keys</div>
          <p className="text-sm text-muted-foreground">
            API keys grant programmatic access. Create keys per application and
            revoke unused keys regularly.
          </p>
          <Button asChild size="sm" variant="outline">
            <Link href="/select-workspace">Go to workspace API keys</Link>
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
        Password reset and session/device management will be added as the auth
        provider abstraction stabilizes.
      </div>
    </div>
  );
}

