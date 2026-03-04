"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useContextAction } from "@/app/context-provider";
import { QueryLakeCreateOrganization, QueryLakeResolveInvitation } from "@/hooks/querylakeAPI";
import { membershipType } from "@/types/globalTypes";

const LAST_WORKSPACE_KEY = "ql_last_workspace";

type WorkspaceCard = {
  name: string;
  slug: string;
  description: string;
  type: "personal" | "organization";
};

export default function Page() {
  const router = useRouter();
  const { userData, authReviewed, loginValid, getUserData } = useContextAction();
  const [memberships, setMemberships] = useState<membershipType[]>([]);
  const [actionInFlight, setActionInFlight] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [retrying, setRetrying] = useState(false);
  const [createOrgOpen, setCreateOrgOpen] = useState(false);
  const [createOrgName, setCreateOrgName] = useState("");
  const [creatingOrg, setCreatingOrg] = useState(false);

  useEffect(() => {
    if (userData?.memberships) {
      setMemberships(userData.memberships);
    }
  }, [userData?.memberships]);

  const { accepted, invites } = useMemo(() => {
    const acceptedMemberships = memberships.filter(
      (membership) => !membership.invite_still_open
    );
    const inviteMemberships = memberships.filter(
      (membership) => membership.invite_still_open
    );
    return { accepted: acceptedMemberships, invites: inviteMemberships };
  }, [memberships]);

  const workspaces: WorkspaceCard[] = useMemo(() => {
    const list: WorkspaceCard[] = [
      {
        name: userData?.username
          ? `${userData.username} (Personal)`
          : "Personal",
        slug: "personal",
        description: "Default workspace for your own collections and runs.",
        type: "personal",
      },
    ];

    accepted.forEach((membership) => {
      list.push({
        name: membership.organization_name,
        slug: membership.organization_id,
        description: `Organization workspace • ${membership.role}`,
        type: "organization",
      });
    });

    return list;
  }, [accepted, userData?.username]);

  const filteredWorkspaces = useMemo(() => {
    if (!filter.trim()) return workspaces;
    const lower = filter.trim().toLowerCase();
    return workspaces.filter((workspace) =>
      workspace.name.toLowerCase().includes(lower)
    );
  }, [workspaces, filter]);

  const storeLastWorkspace = (slug: string) => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LAST_WORKSPACE_KEY, slug);
  };

  const createOrganization = () => {
    if (!userData?.auth) return;
    if (!createOrgName.trim()) {
      toast("Organization name is required");
      return;
    }
    setCreatingOrg(true);
    QueryLakeCreateOrganization({
      auth: userData.auth,
      organization_name: createOrgName.trim(),
      onFinish: (result) => {
        setCreatingOrg(false);
        if (!result) {
          toast("Failed to create organization");
          return;
        }
        toast("Organization created");
        setCreateOrgOpen(false);
        setCreateOrgName("");
        getUserData(userData.auth, () => {
          storeLastWorkspace(result.organization_id);
          router.push(`/w/${result.organization_id}/dashboard`);
        });
      },
    });
  };

  const resolveInvite = (membership: membershipType, accept: boolean) => {
    if (!userData?.auth) return;
    setActionInFlight(membership.organization_id);
    QueryLakeResolveInvitation({
      auth: userData.auth,
      organization_id: membership.organization_id,
      accept,
      onFinish: () => {
        toast(accept ? "Invitation accepted" : "Invitation declined");
        getUserData(userData.auth, () => {
          setActionInFlight(null);
        });
      },
    });
  };

  if (!authReviewed) {
    return (
      <div className="max-w-4xl space-y-6">
        <div className="h-6 w-48 rounded bg-muted" />
        <div className="h-4 w-72 rounded bg-muted" />
        <div className="grid gap-4 md:grid-cols-2">
          <div className="h-32 rounded-lg border border-dashed border-border" />
          <div className="h-32 rounded-lg border border-dashed border-border" />
        </div>
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="max-w-3xl space-y-4">
        <h1 className="text-2xl font-semibold">Select workspace</h1>
        <p className="text-sm text-muted-foreground">
          Please sign in to view your workspaces.
        </p>
        <Button asChild>
          <Link href="/auth/login">Go to login</Link>
        </Button>
        <p className="text-xs text-muted-foreground">
          Auth provider: username/password (local). OAuth providers are not
          enabled yet.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Select workspace</h1>
        <p className="text-sm text-muted-foreground">
          Choose a workspace to open. This list is pulled from your current
          memberships.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Input
          className="w-[260px]"
          placeholder="Filter workspaces"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setFilter("");
          }}
        >
          Clear
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={retrying}
          onClick={() => {
            if (!userData?.auth) return;
            setRetrying(true);
            getUserData(userData.auth, () => setRetrying(false));
          }}
        >
          {retrying ? "Refreshing..." : "Refresh"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            if (typeof window === "undefined") return;
            window.localStorage.removeItem(LAST_WORKSPACE_KEY);
            toast("Cleared last workspace");
          }}
        >
          Clear last workspace
        </Button>
      </div>

      <section className="grid gap-4 md:grid-cols-2">
        {filteredWorkspaces.map((workspace) => (
          <div
            key={workspace.slug}
            className="rounded-lg border border-border bg-card/40 p-5"
          >
            <h2 className="text-base font-semibold">{workspace.name}</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {workspace.description}
            </p>
            <div className="mt-4">
              <Button
                asChild
                size="sm"
                onClick={() => storeLastWorkspace(workspace.slug)}
              >
                <Link href={`/w/${workspace.slug}/dashboard`}>Open</Link>
              </Button>
            </div>
          </div>
        ))}
      </section>

      {invites.length > 0 && (
        <section className="space-y-3 rounded-lg border border-dashed border-border p-5">
          <div>
            <h2 className="text-base font-semibold">Pending invitations</h2>
            <p className="text-sm text-muted-foreground">
              Accept or decline organization invitations.
            </p>
          </div>
          <div className="space-y-3">
            {invites.map((invite) => (
              <div
                key={invite.organization_id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-background px-4 py-3 text-sm"
              >
                <div>
                  <div className="font-medium">{invite.organization_name}</div>
                  <div className="text-muted-foreground">
                    Role: {invite.role}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={actionInFlight === invite.organization_id}
                    onClick={() => resolveInvite(invite, false)}
                  >
                    Decline
                  </Button>
                  <Button
                    size="sm"
                    disabled={actionInFlight === invite.organization_id}
                    onClick={() => resolveInvite(invite, true)}
                  >
                    Accept
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="rounded-lg border border-border bg-card/40 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">Create organization</h2>
            <p className="text-sm text-muted-foreground">
              Create a new organization workspace for your team.
            </p>
          </div>
          <Dialog open={createOrgOpen} onOpenChange={setCreateOrgOpen}>
            <DialogTrigger asChild>
              <Button size="sm" variant="outline">
                New organization
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create organization</DialogTitle>
                <DialogDescription>
                  Pick a name for your organization workspace.
                </DialogDescription>
              </DialogHeader>
              <Input
                placeholder="Organization name"
                value={createOrgName}
                onChange={(event) => setCreateOrgName(event.target.value)}
              />
              <DialogFooter>
                <Button
                  onClick={createOrganization}
                  disabled={!createOrgName.trim() || creatingOrg}
                >
                  {creatingOrg ? "Creating..." : "Create"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </section>
    </div>
  );
}
