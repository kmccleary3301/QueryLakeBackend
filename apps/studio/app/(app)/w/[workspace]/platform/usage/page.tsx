"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useContextAction } from "@/app/context-provider";
import { QuerylakeFetchUsage, UsageEntryType } from "@/hooks/querylakeAPI";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { Skeleton } from "@/components/ui/skeleton";

const isPersonalWorkspace = (workspace: string) =>
  workspace === "personal" || workspace === "me";

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  const { userData, authReviewed, loginValid } = useContextAction();
  const [usage, setUsage] = useState<UsageEntryType[]>([]);
  const [loading, setLoading] = useState(true);
  const [rangeDays, setRangeDays] = useState(30);
  const [window, setWindow] = useState<"hour" | "day" | "month">("day");

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) {
      setLoading(false);
      return;
    }
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - 60 * 60 * 24 * rangeDays;
    setLoading(true);
    QuerylakeFetchUsage({
      auth: userData.auth,
      start_time: startTime,
      end_time: endTime,
      window,
      onFinish: (result) => {
        if (result && Array.isArray(result)) {
          setUsage(result);
        } else {
          setUsage([]);
        }
        setLoading(false);
      },
    });
  }, [authReviewed, loginValid, userData?.auth, rangeDays, window]);

  const scopedUsage = useMemo(() => {
    if (!params.workspace) return usage;
    if (isPersonalWorkspace(params.workspace)) {
      return usage.filter((entry) => entry.organization_id == null);
    }
    return usage.filter((entry) => entry.organization_id === params.workspace);
  }, [params.workspace, usage]);

  const summary = useMemo(() => {
    const apiKeys = new Set<string>();
    const users = new Set<string>();
    const orgs = new Set<string>();
    scopedUsage.forEach((entry) => {
      if (entry.api_key_id) apiKeys.add(entry.api_key_id);
      if (entry.user_id) users.add(entry.user_id);
      if (entry.organization_id) orgs.add(entry.organization_id);
    });
    return {
      totalEntries: scopedUsage.length,
      apiKeys: apiKeys.size,
      users: users.size,
      orgs: orgs.size,
    };
  }, [scopedUsage]);

  const exportUsageJson = () => {
    const exportSuffix = `${params.workspace}-${rangeDays}d-${window}`;
    const blob = new Blob([JSON.stringify(scopedUsage, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `querylake-usage-${exportSuffix}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const exportUsageCsv = () => {
    if (scopedUsage.length === 0) return;
    const exportSuffix = `${params.workspace}-${rangeDays}d-${window}`;
    const rows = scopedUsage.map((entry) => ({
      id: entry.id,
      start_timestamp: entry.start_timestamp,
      start_date: new Date(entry.start_timestamp * 1000).toISOString(),
      window: entry.window,
      api_key_id: entry.api_key_id ?? "",
      organization_id: entry.organization_id ?? "",
      user_id: entry.user_id,
      value: JSON.stringify(entry.value ?? {}),
    }));
    const headers = Object.keys(rows[0]);
    const escapeCell = (value: string) =>
      `"${value.replace(/\"/g, "\"\"")}"`;
    const csv =
      headers.join(",") +
      "\n" +
      rows
        .map((row) =>
          headers
            .map((header) => escapeCell(String(row[header as keyof typeof row])))
            .join(",")
        )
        .join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `querylake-usage-${exportSuffix}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (!authReviewed) {
    return (
      <div className="space-y-4">
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-4 w-44" />
        </div>
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Usage</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view usage metrics.
        </p>
      </div>
    );
  }

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
                <BreadcrumbLink href={`/w/${params.workspace}/platform`}>Platform</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbItem>
                <BreadcrumbPage>Usage</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Usage</h1>
          <p className="text-sm text-muted-foreground">
            Track usage, costs, and rate limits across the workspace.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={String(rangeDays)}
            onValueChange={(value) => setRangeDays(Number(value))}
          >
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7">Last 7 days</SelectItem>
              <SelectItem value="30">Last 30 days</SelectItem>
              <SelectItem value="90">Last 90 days</SelectItem>
              <SelectItem value="365">Last 12 months</SelectItem>
            </SelectContent>
          </Select>
          <Select value={window} onValueChange={(value) => setWindow(value as "hour" | "day" | "month")}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Window" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="hour">Hourly</SelectItem>
              <SelectItem value="day">Daily</SelectItem>
              <SelectItem value="month">Monthly</SelectItem>
          </SelectContent>
          </Select>
          <Button
            variant="outline"
            onClick={exportUsageJson}
            disabled={scopedUsage.length === 0}
          >
            Export JSON
          </Button>
          <Button
            variant="outline"
            onClick={exportUsageCsv}
            disabled={scopedUsage.length === 0}
          >
            Export CSV
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Entries</div>
          <div className="mt-1 text-2xl font-semibold">
            {loading ? "—" : summary.totalEntries}
          </div>
        </div>
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Active API keys</div>
          <div className="mt-1 text-2xl font-semibold">
            {loading ? "—" : summary.apiKeys}
          </div>
        </div>
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Active users</div>
          <div className="mt-1 text-2xl font-semibold">
            {loading ? "—" : summary.users}
          </div>
        </div>
        <div className="rounded-lg border border-border p-4 text-sm">
          <div className="text-xs text-muted-foreground">Active orgs</div>
          <div className="mt-1 text-2xl font-semibold">
            {loading ? "—" : summary.orgs}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
        {loading
          ? `Loading usage for the last ${rangeDays} days...`
          : `Found ${summary.totalEntries} usage entries in the last ${rangeDays} days.`}
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Window start</TableHead>
              <TableHead>Window</TableHead>
              <TableHead>API key</TableHead>
              <TableHead>Organization</TableHead>
              <TableHead>Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} className="py-6 text-center text-sm text-muted-foreground">
                  Loading usage entries...
                </TableCell>
              </TableRow>
            ) : scopedUsage.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="py-6 text-center text-sm text-muted-foreground">
                  No usage entries available.
                </TableCell>
              </TableRow>
            ) : (
              scopedUsage.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell>
                    {new Date(entry.start_timestamp * 1000).toLocaleDateString()}
                  </TableCell>
                  <TableCell>{entry.window}</TableCell>
                  <TableCell>{entry.api_key_id ?? "—"}</TableCell>
                  <TableCell>{entry.organization_id ?? "—"}</TableCell>
                  <TableCell className="max-w-[240px] truncate font-mono text-xs text-muted-foreground">
                    {JSON.stringify(entry.value ?? {})}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
