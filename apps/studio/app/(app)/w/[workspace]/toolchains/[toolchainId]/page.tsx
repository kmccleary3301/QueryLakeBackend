"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import RuntimeModeBanner from "@/components/toolchains/runtime-mode-banner";
import { useRuntimeMode } from "@/components/toolchains/runtime-mode";
import { useContextAction } from "@/app/context-provider";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { fetchToolchainConfig } from "@/hooks/querylakeAPI";
import { toolchain_session } from "@/types/globalTypes";
import { ToolChain } from "@/types/toolchains";

export default function ToolchainPage() {
  const params = useParams<{ workspace: string; toolchainId: string }>()!;
  const router = useRouter();
  const {
    userData,
    setSelectedToolchain,
    toolchainSessions,
    refreshToolchainSessions,
    authReviewed,
    loginValid,
  } = useContextAction();
  const { mode } = useRuntimeMode();
  const [toolchain, setToolchain] = useState<ToolChain | null>(null);
  const [loading, setLoading] = useState(true);
  const [creatingSession, setCreatingSession] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) {
      setLoading(false);
      return;
    }
    setLoading(true);
    fetchToolchainConfig({
      auth: userData.auth,
      toolchain_id: params.toolchainId,
      onFinish: (result: ToolChain) => {
        setToolchain(result);
        setLoading(false);
      },
    });
  }, [authReviewed, loginValid, userData?.auth, params.toolchainId]);

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

  const nodePreview = useMemo(() => {
    return toolchain?.nodes?.slice(0, 6) ?? [];
  }, [toolchain?.nodes]);

  const recentRuns: toolchain_session[] = useMemo(() => {
    const runs = Array.from(toolchainSessions.values()).filter(
      (run) => run.toolchain === params.toolchainId
    );
    return runs.sort((a, b) => b.time - a.time).slice(0, 10);
  }, [params.toolchainId, toolchainSessions]);

  const openLegacyRunner = () => {
    setSelectedToolchain(params.toolchainId);
    router.push("/app/create");
  };

  const createV2Session = async () => {
    if (!userData?.auth) return;
    setCreatingSession(true);
    setCreateError(null);
    try {
      const response = await fetch("/v2/kernel/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${userData.auth}`,
        },
        body: JSON.stringify({
          toolchain_id: params.toolchainId,
          title: toolchain?.name,
        }),
      });
      if (!response.ok) {
        setCreateError(`Failed to create session (${response.status}).`);
        setCreatingSession(false);
        return;
      }
      const data = await response.json();
      const sessionId = data?.session_id;
      if (!sessionId) {
        setCreateError("Session created but no session_id returned.");
        setCreatingSession(false);
        return;
      }
      router.push(`/w/${params.workspace}/runs/${sessionId}`);
    } catch (error) {
      setCreateError(`Failed to create session: ${String(error)}`);
      setCreatingSession(false);
    }
  };

  return (
    <div className="space-y-6">
      <RuntimeModeBanner />
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbLink href={`/w/${params.workspace}/toolchains`}>
                  Toolchains
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>{params.toolchainId}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">
            Toolchain: {toolchain?.name ?? params.toolchainId}
          </h1>
          <p className="text-sm text-muted-foreground">
            Workspace {params.workspace}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline">
            <Link href={`/w/${params.workspace}/toolchains`}>Back to toolchains</Link>
          </Button>
          <Button onClick={openLegacyRunner} variant="outline">
            Open legacy runner
          </Button>
          {mode === "v2" && (
            <Button onClick={createV2Session} disabled={creatingSession}>
              {creatingSession ? "Creating session..." : "Create v2 session"}
            </Button>
          )}
          <Button asChild variant="outline">
            <Link href="/nodes/node_editor">Open legacy builder</Link>
          </Button>
        </div>
      </div>
      {createError ? (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {createError}
        </div>
      ) : null}

      {loading ? (
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-52" />
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-4 w-44" />
        </div>
      ) : !toolchain ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Unable to load toolchain details. Check your auth and backend status.
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <div className="space-y-4 rounded-lg border border-border p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">Overview</h2>
              <span className="text-xs text-muted-foreground">
                Runtime: {mode === "v2" ? "v2 sessions" : "v1 legacy"}
              </span>
            </div>
            <div className="grid gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">ID</span>
                <span className="font-medium">{toolchain.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Category</span>
                <span className="font-medium">{toolchain.category}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Nodes</span>
                <span className="font-medium">
                  {toolchain.nodes?.length ?? 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">First event</span>
                <span className="font-medium">
                  {toolchain.first_event_follow_up ?? "—"}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border p-5 text-sm">
            <div className="font-semibold">Runtime notes</div>
            <p className="mt-2 text-muted-foreground">
              {mode === "v2"
                ? "v2 sessions are not wired to the new UI yet. Use the legacy runner until the SSE pipeline is hooked up."
                : "Legacy WebSocket runtime is active. Sessions will open in /app/create."}
            </p>
          </div>
        </div>
      )}

      {toolchain && (
        <div className="rounded-lg border border-border p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold">Node preview</h2>
            <span className="text-xs text-muted-foreground">
              Showing {nodePreview.length} of {toolchain.nodes.length}
            </span>
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {nodePreview.map((node) => (
              <div
                key={node.id}
                className="rounded-md border border-border bg-background px-4 py-3 text-sm"
              >
                <div className="font-medium">{node.id}</div>
                <div className="text-xs text-muted-foreground">
                  {node.api_function ?? "Custom node"}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="rounded-lg border border-border p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold">Recent runs</h2>
          <div className="flex flex-wrap items-center gap-2">
            <Button asChild size="sm">
              <Link href={`/w/${params.workspace}/runs/new?toolchain=${params.toolchainId}`}>
                New run
              </Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/runs`}>View all runs</Link>
            </Button>
          </div>
        </div>
        {recentRuns.length === 0 ? (
          <div className="mt-3 rounded-md border border-dashed border-border p-4 text-sm text-muted-foreground">
            No runs recorded for this toolchain yet.
          </div>
        ) : (
          <div className="mt-3 divide-y divide-border rounded-md border border-border">
            {recentRuns.map((run) => (
              <div
                key={run.id}
                className="flex flex-wrap items-center justify-between gap-4 px-4 py-3 text-sm"
              >
                <div>
                  <div className="font-medium">{run.title}</div>
                  <div className="text-xs text-muted-foreground">{run.id}</div>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>{new Date(run.time * 1000).toLocaleString()}</span>
                  <Button asChild size="sm" variant="outline">
                    <Link href={`/w/${params.workspace}/runs/${run.id}`}>Open</Link>
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
