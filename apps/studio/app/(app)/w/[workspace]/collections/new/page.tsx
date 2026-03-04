"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { useContextAction } from "@/app/context-provider";
import { createCollection } from "@/hooks/querylakeAPI";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

export default function Page() {
  const router = useRouter();
  const params = useParams<{ workspace: string }>()!;
  const { userData } = useContextAction();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [isPublic, setIsPublic] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ownerLabel = useMemo(() => {
    if (isPersonalWorkspace(params.workspace)) return "Personal workspace";
    const membership = userData?.memberships.find(
      (member) => member.organization_id === params.workspace
    );
    return membership?.organization_name ?? "Organization workspace";
  }, [params.workspace, userData?.memberships]);

  const submit = () => {
    if (!userData?.auth) return;
    if (!title.trim()) {
      setError("Collection name is required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    createCollection({
      auth: userData.auth,
      title: title.trim(),
      description: description.trim(),
      public: isPublic,
      ...(isPersonalWorkspace(params.workspace)
        ? {}
        : { organization_id: params.workspace }),
      onFinish: (result) => {
        setSubmitting(false);
        if (!result) {
          setError("Failed to create collection.");
          return;
        }
        router.push(`/w/${params.workspace}/collections/${result.hash_id}`);
      },
    });
  };

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}/collections`}>
                Collections
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>New</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">New collection</h1>
        <p className="text-sm text-muted-foreground">
          Create a collection with indexing and parsing settings.
        </p>
        <div className="mt-3">
          <Button asChild size="sm" variant="outline">
            <Link href={`/w/${params.workspace}/collections`}>
              Back to collections
            </Link>
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-border p-5 space-y-4">
        <div>
          <div className="text-sm font-medium">Name</div>
          <Input
            className="mt-2"
            placeholder="Collection name"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
          />
        </div>
        <div>
          <div className="text-sm font-medium">Description</div>
          <Textarea
            className="mt-2"
            placeholder="What will this collection contain?"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>
        <div className="flex items-center justify-between rounded-md border border-border px-4 py-3">
          <div>
            <div className="text-sm font-medium">Public collection</div>
            <div className="text-xs text-muted-foreground">
              Allow other workspace members to discover this collection.
            </div>
          </div>
          <Switch checked={isPublic} onCheckedChange={setIsPublic} />
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Owner: {ownerLabel}</span>
          {error ? <span className="text-destructive">{error}</span> : null}
        </div>
        <Button disabled={submitting} onClick={submit}>
          {submitting ? "Creating..." : "Create collection"}
        </Button>
      </div>
    </div>
  );
}
