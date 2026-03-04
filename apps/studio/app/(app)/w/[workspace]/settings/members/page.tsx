"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useContextAction } from "@/app/context-provider";
import {
  QueryLakeFetchOrganizationsMemberships,
  QueryLakeFetchUsersMemberships,
  QueryLakeInviteUserToOrg,
  QueryLakeResolveInvitation,
  QueryLakeUpdateOrgMemberRole,
  memberRoleLower,
  organization_memberships,
  user_organization_membership,
} from "@/hooks/querylakeAPI";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

const roleOptions: memberRoleLower[] = ["owner", "admin", "member", "viewer"];
const normalizeRole = (role: string): memberRoleLower => {
  if (role.toLowerCase() === "reader") return "viewer";
  const lower = role.toLowerCase() as memberRoleLower;
  return roleOptions.includes(lower) ? lower : "member";
};

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  const { userData, authReviewed, loginValid } = useContextAction();
  const [members, setMembers] = useState<organization_memberships[]>([]);
  const [pendingInvites, setPendingInvites] = useState<
    user_organization_membership[]
  >([]);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<memberRoleLower>("member");
  const [inviteLoading, setInviteLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roleDrafts, setRoleDrafts] = useState<Record<string, memberRoleLower>>(
    {}
  );
  const [roleSaving, setRoleSaving] = useState<string | null>(null);
  const [resolvingInvite, setResolvingInvite] = useState<string | null>(null);

  const refreshInvites = useCallback(() => {
    if (!userData?.auth) return;
    QueryLakeFetchUsersMemberships({
      auth: userData.auth,
      onFinish: (result) => {
        if (result && Array.isArray(result)) {
          setPendingInvites(result.filter((entry) => entry.invite_still_open));
        }
      },
    });
  }, [userData?.auth]);

  const refreshMembers = useCallback(() => {
    if (!userData?.auth) return;
    if (isPersonalWorkspace(params.workspace)) {
      setMembers([]);
      return;
    }
    QueryLakeFetchOrganizationsMemberships({
      auth: userData.auth,
      organization_id: params.workspace,
      onFinish: (result) => {
        if (result && Array.isArray(result)) {
          setMembers(result);
        } else {
          setMembers([]);
        }
      },
    });
  }, [params.workspace, userData?.auth]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    refreshInvites();
  }, [authReviewed, loginValid, userData?.auth, refreshInvites]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    refreshMembers();
  }, [authReviewed, loginValid, userData?.auth, params.workspace, refreshMembers]);

  const canInvite = useMemo(() => {
    if (!userData?.memberships) return false;
    const membership = userData.memberships.find(
      (entry) => entry.organization_id === params.workspace
    );
    return membership?.role === "owner" || membership?.role === "admin";
  }, [userData?.memberships, params.workspace]);

  const invite = () => {
    if (!userData?.auth) return;
    if (!inviteEmail.trim()) {
      setError("Username or email is required.");
      return;
    }
    const target = inviteEmail.trim();
    setInviteLoading(true);
    setError(null);
    QueryLakeInviteUserToOrg({
      auth: userData.auth,
      organization_id: params.workspace,
      username: target,
      role: inviteRole,
      onFinish: (success) => {
        setInviteLoading(false);
        if (!success) {
          setError("Invite failed.");
          toast("Invite failed. Check the username and permissions.");
          return;
        }
        setInviteEmail("");
        toast(`Invite sent to ${target}.`);
      },
    });
  };

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
        <h1 className="text-2xl font-semibold">Members</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to manage workspace members.
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
              <BreadcrumbPage>Members</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Members</h1>
        <p className="text-sm text-muted-foreground">
          Invite, role-manage, and audit workspace members.
        </p>
      </div>

      {isPersonalWorkspace(params.workspace) ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          <p>
            Personal workspaces do not have members. Switch to an organization
            workspace to manage team access.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button asChild size="sm" variant="outline">
              <Link href="/select-workspace">Select workspace</Link>
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="rounded-lg border border-border p-5 space-y-4">
            <div className="text-sm font-semibold">Invite member</div>
            <div className="flex flex-wrap gap-3">
              <Input
                className="min-w-[240px]"
                placeholder="Username or email"
                value={inviteEmail}
                onChange={(event) => setInviteEmail(event.target.value)}
              />
              <Select
                value={inviteRole}
                onValueChange={(value) => setInviteRole(value as memberRoleLower)}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Role" />
                </SelectTrigger>
                <SelectContent>
                  {roleOptions.map((role) => (
                    <SelectItem key={role} value={role}>
                      {role}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button disabled={!canInvite || inviteLoading} onClick={invite}>
                {inviteLoading ? "Inviting..." : "Send invite"}
              </Button>
            </div>
            {!canInvite && (
              <p className="text-xs text-muted-foreground">
                Only org owners/admins can invite new members.
              </p>
            )}
            {error ? <p className="text-xs text-destructive">{error}</p> : null}
          </div>

          <div className="rounded-lg border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Member</TableHead>
                  <TableHead>Role</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {members.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={2} className="py-6 text-center text-sm text-muted-foreground">
                      No members found for this organization.
                    </TableCell>
                  </TableRow>
                ) : (
                  members.map((member) => (
                    <TableRow key={`${member.username}-${member.organization_id}`}>
                      <TableCell className="font-medium">{member.username}</TableCell>
                      <TableCell>
                        <div className="flex flex-wrap items-center gap-2">
                          <Select
                            value={
                              roleDrafts[member.username] ??
                              normalizeRole(member.role)
                            }
                            onValueChange={(value) =>
                              setRoleDrafts((prev) => ({
                                ...prev,
                                [member.username]: value as memberRoleLower,
                              }))
                            }
                            disabled={!canInvite || member.username === userData.username}
                          >
                            <SelectTrigger className="w-[160px]">
                              <SelectValue placeholder="Role" />
                            </SelectTrigger>
                            <SelectContent>
                              {roleOptions.map((role) => (
                                <SelectItem key={role} value={role}>
                                  {role.charAt(0).toUpperCase() + role.slice(1)}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={
                              !canInvite ||
                              member.username === userData.username ||
                              roleSaving === member.username ||
                              (roleDrafts[member.username] ?? normalizeRole(member.role)) ===
                                normalizeRole(member.role)
                            }
                            onClick={() => {
                              if (!userData?.auth) return;
                              const nextRole =
                                roleDrafts[member.username] ??
                                normalizeRole(member.role);
                              setRoleSaving(member.username);
                              QueryLakeUpdateOrgMemberRole({
                                auth: userData.auth,
                                organization_id: params.workspace,
                                username: member.username,
                                role: nextRole,
                                onFinish: (success) => {
                                  setRoleSaving(null);
                                  if (!success) {
                                    toast("Failed to update member role.");
                                    setRoleDrafts((prev) => ({
                                      ...prev,
                                      [member.username]: normalizeRole(member.role),
                                    }));
                                    return;
                                  }
                                  toast("Member role updated.");
                                  refreshMembers();
                                },
                              });
                            }}
                          >
                            {roleSaving === member.username ? "Saving..." : "Save"}
                          </Button>
                          {member.username === userData.username && (
                            <span className="text-xs text-muted-foreground">
                              You cannot change your own role.
                            </span>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </>
      )}

      {pendingInvites.length > 0 && (
        <div className="rounded-lg border border-border p-5">
          <div className="text-sm font-semibold">Your pending invites</div>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {pendingInvites.map((invite) => (
              <div
                key={invite.organization_id}
                className="rounded-md border border-border bg-background px-4 py-3 text-sm"
              >
                <div className="font-medium">{invite.organization_name}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  Role: {invite.role} • Invited by {invite.sender}
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={resolvingInvite === invite.organization_id}
                    onClick={() => {
                      if (!userData?.auth) return;
                      setResolvingInvite(invite.organization_id);
                      QueryLakeResolveInvitation({
                        auth: userData.auth,
                        organization_id: invite.organization_id,
                        accept: false,
                        onFinish: (success) => {
                          setResolvingInvite(null);
                          toast(success ? "Invitation declined" : "Failed to decline invitation");
                          refreshInvites();
                        },
                      });
                    }}
                  >
                    Decline
                  </Button>
                  <Button
                    size="sm"
                    disabled={resolvingInvite === invite.organization_id}
                    onClick={() => {
                      if (!userData?.auth) return;
                      setResolvingInvite(invite.organization_id);
                      QueryLakeResolveInvitation({
                        auth: userData.auth,
                        organization_id: invite.organization_id,
                        accept: true,
                        onFinish: (success) => {
                          setResolvingInvite(null);
                          toast(success ? "Invitation accepted" : "Failed to accept invitation");
                          refreshInvites();
                        },
                      });
                    }}
                  >
                    Accept
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
