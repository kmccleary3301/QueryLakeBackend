"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { useContextAction } from "@/app/context-provider";
import { QuerylakeFetchUsage, UsageEntryType } from "@/hooks/querylakeAPI";
import { collectionGroup } from "@/types/globalTypes";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

const quickLinks = [
  {
    title: "Collections",
    description: "Organize knowledge bases and retrieval-ready corpora.",
    href: "collections",
  },
  {
    title: "Files",
    description: "Upload and parse documents into your collections.",
    href: "files",
  },
  {
    title: "Toolchains",
    description: "Build and version workflow graphs for your apps.",
    href: "toolchains",
  },
  {
    title: "Runs",
    description: "Monitor live runs and replay historical sessions.",
    href: "runs",
  },
];

export default function DashboardPage() {
  const params = useParams<{ workspace: string }>()!;
  const workspace = params.workspace;
  const {
    userData,
    authReviewed,
    loginValid,
    collectionGroups,
    refreshCollectionGroups,
    toolchainSessions,
    refreshToolchainSessions,
  } = useContextAction();

  const [usage, setUsage] = useState<UsageEntryType[] | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (collectionGroups.length === 0) {
      refreshCollectionGroups();
    }
  }, [
    authReviewed,
    loginValid,
    userData?.auth,
    collectionGroups.length,
    refreshCollectionGroups,
  ]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (toolchainSessions.size === 0) {
      refreshToolchainSessions();
    }
  }, [
    authReviewed,
    loginValid,
    userData?.auth,
    toolchainSessions.size,
    refreshToolchainSessions,
  ]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) {
      setUsage(null);
      setUsageLoading(false);
      return;
    }

    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - 60 * 60 * 24 * 30;
    setUsageLoading(true);

    QuerylakeFetchUsage({
      auth: userData.auth,
      start_time: startTime,
      end_time: endTime,
      window: "day",
      onFinish: (result) => {
        if (result && Array.isArray(result)) {
          setUsage(result);
        } else {
          setUsage([]);
        }
        setUsageLoading(false);
      },
    });
  }, [authReviewed, loginValid, userData?.auth]);

  const filteredGroups: collectionGroup[] = useMemo(() => {
    if (collectionGroups.length === 0) return [];
    if (isPersonalWorkspace(workspace)) {
      return collectionGroups.filter((group) =>
        ["My Collections", "Global Collections"].includes(group.title)
      );
    }
    const membership = userData?.memberships.find(
      (member) => member.organization_id === workspace
    );
    if (!membership) return collectionGroups;
    const match = collectionGroups.filter(
      (group) => group.title === membership.organization_name
    );
    return match.length ? match : collectionGroups;
  }, [collectionGroups, userData?.memberships, workspace]);

  const collectionsCount = useMemo(() => {
    const groupsToCount = filteredGroups.length ? filteredGroups : collectionGroups;
    return groupsToCount.reduce((acc, group) => acc + group.collections.length, 0);
  }, [collectionGroups, filteredGroups]);

  const usageEntriesCount = useMemo(() => {
    if (!usage || usage.length === 0) return 0;
    if (isPersonalWorkspace(workspace)) {
      return usage.filter((entry) => entry.organization_id == null).length;
    }
    return usage.filter((entry) => entry.organization_id === workspace).length;
  }, [usage, workspace]);

  const runsCount = toolchainSessions.size;

  if (!authReviewed) {
    return (
      <div className="space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-5 w-52" />
          <Skeleton className="h-4 w-72" />
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Workspace dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view your workspace overview.
        </p>
        <Button asChild size="sm">
          <Link href="/auth/login">Go to login</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${workspace}`}>Workspace</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>Dashboard</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">Workspace dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Overview of collections, runs, and usage for this workspace.
        </p>
      </div>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Collections</div>
          <div className="mt-1 text-2xl font-semibold">{collectionsCount}</div>
          <div className="mt-3">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/collections`}>View collections</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Runs</div>
          <div className="mt-1 text-2xl font-semibold">{runsCount}</div>
          <div className="mt-3">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/runs`}>View runs</Link>
            </Button>
          </div>
        </div>
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">
            Usage entries (30d)
          </div>
          <div className="mt-1 text-2xl font-semibold">
            {usageLoading ? "—" : usageEntriesCount}
          </div>
          <div className="mt-3">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${workspace}/platform/usage`}>View usage</Link>
            </Button>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {quickLinks.map((link) => (
          <div
            key={link.title}
            className="rounded-lg border border-border bg-card/40 p-5"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold">{link.title}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {link.description}
                </p>
              </div>
              <Button asChild size="sm" variant="outline">
                <Link href={`/w/${workspace}/${link.href}`}>Open</Link>
              </Button>
            </div>
          </div>
        ))}
      </section>

      <section className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
        This dashboard will grow into an activity feed (uploads, ingestion,
        run/job status) as the v2 runtime and Files pipelines get adopted.
      </section>
    </div>
  );
}

