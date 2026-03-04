"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import RuntimeModeBanner from "@/components/toolchains/runtime-mode-banner";
import { useRuntimeMode } from "@/components/toolchains/runtime-mode";
import { useContextAction } from "@/app/context-provider";
import { toolchain_session } from "@/types/globalTypes";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  const { mode } = useRuntimeMode();
  const {
    userData,
    toolchainSessions,
    refreshToolchainSessions,
    authReviewed,
    loginValid,
  } = useContextAction();
  const [searchQuery, setSearchQuery] = useState("");
  const [toolchainFilter, setToolchainFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

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

  const runs: toolchain_session[] = useMemo(() => {
    const list = Array.from(toolchainSessions.values());
    return list.sort((a, b) => b.time - a.time);
  }, [toolchainSessions]);

  const toolchainOptions = useMemo(() => {
    const options = new Set<string>();
    runs.forEach((run) => {
      if (run.toolchain) options.add(run.toolchain);
    });
    return ["all", ...Array.from(options).sort()];
  }, [runs]);

  const deriveStatus = (run: toolchain_session) => {
    const ageSeconds = Date.now() / 1000 - run.time;
    if (ageSeconds < 60 * 60) return "active";
    if (ageSeconds < 60 * 60 * 24) return "recent";
    return "archived";
  };

  const filteredRuns = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return runs.filter((run) => {
      if (toolchainFilter !== "all" && run.toolchain !== toolchainFilter) {
        return false;
      }
      if (statusFilter !== "all" && deriveStatus(run) !== statusFilter) {
        return false;
      }
      if (!query) return true;
      return (
        run.title.toLowerCase().includes(query) ||
        run.toolchain.toLowerCase().includes(query) ||
        run.id.toLowerCase().includes(query)
      );
    });
  }, [runs, searchQuery, toolchainFilter, statusFilter]);

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
              <BreadcrumbItem>
                <BreadcrumbPage>Runs</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Runs</h1>
          <p className="text-sm text-muted-foreground">
            Observe toolchain runs and live session outputs.
          </p>
        </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              onClick={() => refreshToolchainSessions()}
              disabled={!authReviewed || !loginValid}
            >
              Refresh
            </Button>
            <Button asChild variant="outline">
              <Link href={`/w/${params.workspace}/runs/new`}>New run</Link>
            </Button>
          </div>
        </div>

      {mode === "v2" && (
        <div className="rounded-lg border border-dashed border-border p-4 text-xs text-muted-foreground">
          v2 sessions are listed here via the same legacy session index as v1 runs.
          If a session is missing, create a new one from the “New run” page or open
          it directly by ID.
        </div>
      )}

      {!authReviewed ? (
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-48" />
        </div>
      ) : !loginValid || !userData ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Sign in to view run history.
        </div>
      ) : runs.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          No runs recorded yet.
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Input
              className="w-[240px]"
              placeholder="Search runs..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
            />
            <Select value={toolchainFilter} onValueChange={setToolchainFilter}>
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Filter by toolchain" />
              </SelectTrigger>
              <SelectContent>
                {toolchainOptions.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option === "all" ? "All toolchains" : option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="recent">Recent</SelectItem>
                <SelectItem value="archived">Archived</SelectItem>
              </SelectContent>
            </Select>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setSearchQuery("");
                setToolchainFilter("all");
                setStatusFilter("all");
              }}
            >
              Clear filters
            </Button>
          </div>
          <div className="text-xs text-muted-foreground">
            Status is derived from the run timestamp (active = last hour, recent = last 24h).
          </div>

          {filteredRuns.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
              No runs match your filters.
            </div>
          ) : (
            <div className="rounded-lg border border-border">
              <div className="divide-y divide-border">
                {filteredRuns.map((run) => (
                  <div
                    key={run.id}
                    className="flex flex-wrap items-center justify-between gap-4 px-5 py-4 text-sm"
                  >
                    <div>
                      <div className="font-medium">{run.title}</div>
                      <div className="text-xs text-muted-foreground">
                        Toolchain: {run.toolchain}
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {run.id}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <Badge variant="secondary" className="capitalize">
                        {deriveStatus(run)}
                      </Badge>
                      <span>{new Date(run.time * 1000).toLocaleString()}</span>
                      <Button asChild size="sm" variant="outline">
                        <Link href={`/w/${params.workspace}/runs/${run.id}`}>
                          Open
                        </Link>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
