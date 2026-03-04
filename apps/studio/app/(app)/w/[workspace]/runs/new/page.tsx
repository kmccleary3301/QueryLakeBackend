"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import RuntimeModeBanner from "@/components/toolchains/runtime-mode-banner";
import { useRuntimeMode } from "@/components/toolchains/runtime-mode";
import { useContextAction } from "@/app/context-provider";
import { toolchain_type } from "@/types/globalTypes";

type FlattenedToolchain = toolchain_type & { categoryLabel?: string };

export default function NewRunPage() {
  const params = useParams<{ workspace: string }>()!;
  const router = useRouter();
  const searchParams = useSearchParams();
  const { mode } = useRuntimeMode();
  const { userData, setSelectedToolchain, authReviewed, loginValid } =
    useContextAction();

  const [selectedToolchainId, setSelectedToolchainId] = useState<string>("");
  const [creatingSession, setCreatingSession] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toolchains: FlattenedToolchain[] = useMemo(() => {
    const categories = userData?.available_toolchains ?? [];
    const flattened: FlattenedToolchain[] = [];
    categories.forEach((category) => {
      category.entries.forEach((entry) => {
        flattened.push({ ...entry, categoryLabel: category.category });
      });
    });
    return flattened.sort((a, b) => a.title.localeCompare(b.title));
  }, [userData?.available_toolchains]);

  useEffect(() => {
    if (selectedToolchainId) return;
    const toolchainFromUrl = searchParams?.get("toolchain");
    if (toolchainFromUrl && toolchains.some((entry) => entry.id === toolchainFromUrl)) {
      setSelectedToolchainId(toolchainFromUrl);
      return;
    }
    if (userData?.default_toolchain?.id) {
      setSelectedToolchainId(userData.default_toolchain.id);
    }
  }, [
    searchParams,
    selectedToolchainId,
    toolchains,
    userData?.default_toolchain?.id,
  ]);

  const selectedToolchain = useMemo(() => {
    if (!selectedToolchainId) return null;
    return toolchains.find((entry) => entry.id === selectedToolchainId) ?? null;
  }, [selectedToolchainId, toolchains]);

  const openLegacyRunner = () => {
    if (!selectedToolchainId) return;
    setSelectedToolchain(selectedToolchainId);
    router.push("/app/create");
  };

  const createV2Session = async () => {
    if (!userData?.auth || !selectedToolchainId) return;
    setCreatingSession(true);
    setError(null);
    try {
      const response = await fetch("/v2/kernel/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${userData.auth}`,
        },
        body: JSON.stringify({
          toolchain_id: selectedToolchainId,
          title: selectedToolchain?.title ?? undefined,
        }),
      });
      if (!response.ok) {
        setError(`Failed to create session (${response.status}).`);
        setCreatingSession(false);
        return;
      }
      const data = await response.json();
      const sessionId = data?.session_id;
      if (!sessionId) {
        setError("Session created but no session_id returned.");
        setCreatingSession(false);
        return;
      }
      router.push(`/w/${params.workspace}/runs/${sessionId}`);
    } catch (fetchError) {
      setError(`Failed to create session: ${String(fetchError)}`);
      setCreatingSession(false);
    }
  };

  if (!authReviewed) {
    return (
      <div className="space-y-4">
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-52" />
        </div>
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">New run</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to start a new toolchain run.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-3xl space-y-6">
      <RuntimeModeBanner />

      <div>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbLink href={`/w/${params.workspace}/runs`}>Runs</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>New</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <h1 className="text-2xl font-semibold">New run</h1>
        <p className="text-sm text-muted-foreground">
          Pick a toolchain and start a new run using the selected runtime mode.
        </p>
      </div>

      {error ? (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="rounded-lg border border-border bg-card/40 p-5 space-y-4">
        <div className="grid gap-2">
          <div className="text-sm font-medium">Toolchain</div>
          <Select value={selectedToolchainId} onValueChange={setSelectedToolchainId}>
            <SelectTrigger className="w-full max-w-md">
              <SelectValue placeholder="Select a toolchain" />
            </SelectTrigger>
            <SelectContent>
              {toolchains.map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  {entry.title}
                  {entry.categoryLabel ? ` (${entry.categoryLabel})` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="text-xs text-muted-foreground">
            Runtime mode: {mode === "v2" ? "v2 sessions" : "v1 legacy"}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {mode === "v2" ? (
            <Button
              onClick={createV2Session}
              disabled={!selectedToolchainId || creatingSession}
            >
              {creatingSession ? "Creating session..." : "Create v2 session"}
            </Button>
          ) : (
            <Button onClick={openLegacyRunner} disabled={!selectedToolchainId}>
              Open legacy runner
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => router.push(`/w/${params.workspace}/toolchains`)}
          >
            Browse toolchains
          </Button>
        </div>
      </div>
    </div>
  );
}
