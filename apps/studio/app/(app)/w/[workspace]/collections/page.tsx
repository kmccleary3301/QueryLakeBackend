"use client";

import Link from "next/link";
import { useEffect, useMemo } from "react";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useContextAction } from "@/app/context-provider";
import { collectionGroup } from "@/types/globalTypes";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  const { userData, collectionGroups, refreshCollectionGroups, authReviewed, loginValid } =
    useContextAction();

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (collectionGroups.length === 0) {
      refreshCollectionGroups();
    }
  }, [authReviewed, loginValid, userData?.auth, collectionGroups.length, refreshCollectionGroups]);

  const filteredGroups: collectionGroup[] = useMemo(() => {
    if (collectionGroups.length === 0) return [];
    if (isPersonalWorkspace(params.workspace)) {
      return collectionGroups.filter((group) =>
        ["My Collections", "Global Collections"].includes(group.title)
      );
    }
    const membership = userData?.memberships.find(
      (member) => member.organization_id === params.workspace
    );
    if (!membership) return collectionGroups;
    const match = collectionGroups.filter(
      (group) => group.title === membership.organization_name
    );
    return match.length ? match : collectionGroups;
  }, [collectionGroups, params.workspace, userData?.memberships]);

  const groupsToShow = filteredGroups.length ? filteredGroups : collectionGroups;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbItem>
                <BreadcrumbPage>Collections</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Collections</h1>
          <p className="text-sm text-muted-foreground">
            Manage knowledge bases and retrieval-ready collections.
          </p>
        </div>
        <Button asChild>
          <Link href={`/w/${params.workspace}/collections/new`}>
            New collection
          </Link>
        </Button>
      </div>

      {!authReviewed ? (
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-52" />
        </div>
      ) : !loginValid || !userData ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Sign in to view collections.
        </div>
      ) : groupsToShow.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          <div>No collections loaded yet. Upload a file or create a collection to get started.</div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/collections/new`}>
                Create collection
              </Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/files`}>
                Upload files
              </Link>
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {groupsToShow.map((group) => (
            <div key={group.title} className="rounded-lg border border-border p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">{group.title}</h2>
                <span className="text-xs text-muted-foreground">
                  {group.collections.length} collections
                </span>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {group.collections.map((collection) => (
                  <div
                    key={collection.hash_id}
                    className="rounded-md border border-border bg-background px-4 py-3"
                  >
                    <div className="text-sm font-medium">{collection.title}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {collection.items ?? 0} documents • {collection.type}
                    </div>
                    <div className="mt-3">
                      <Button asChild size="sm" variant="outline">
                        <Link
                          href={`/w/${params.workspace}/collections/${collection.hash_id}`}
                        >
                          Open
                        </Link>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
