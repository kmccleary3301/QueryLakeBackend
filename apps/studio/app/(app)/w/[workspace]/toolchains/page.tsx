"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
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
import { Skeleton } from "@/components/ui/skeleton";
import RuntimeModeBanner from "@/components/toolchains/runtime-mode-banner";
import { useContextAction } from "@/app/context-provider";
import { toolchainCategory } from "@/types/globalTypes";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

export default function Page() {
  const params = useParams<{ workspace: string }>()!;
  const { userData, authReviewed, loginValid } = useContextAction();
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const categories: toolchainCategory[] = useMemo(() => {
    return userData?.available_toolchains ?? [];
  }, [userData?.available_toolchains]);

  const categoryOptions = useMemo(() => {
    const options = categories.map((category) => category.category).sort();
    return ["all", ...options];
  }, [categories]);

  const filteredCategories = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return categories
      .filter(
        (category) => categoryFilter === "all" || category.category === categoryFilter
      )
      .map((category) => ({
        ...category,
        entries: category.entries.filter((toolchain) => {
          if (!query) return true;
          return (
            toolchain.title.toLowerCase().includes(query) ||
            toolchain.id.toLowerCase().includes(query) ||
            toolchain.category.toLowerCase().includes(query)
          );
        }),
      }))
      .filter((category) => category.entries.length > 0);
  }, [categories, categoryFilter, searchQuery]);

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
                <BreadcrumbPage>Toolchains</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Toolchains</h1>
          <p className="text-sm text-muted-foreground">
            Build and manage toolchains for automation and workflows.
          </p>
        </div>
        <Button asChild>
          <Link href="/nodes/node_editor">Create toolchain (legacy builder)</Link>
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
          Sign in to view your toolchains.
        </div>
      ) : categories.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          <div>No toolchains found yet. Create your first toolchain to get started.</div>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href="/nodes/node_editor">Create toolchain (legacy builder)</Link>
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Input
              className="w-[260px]"
              placeholder="Search toolchains..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
            />
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Filter by category" />
              </SelectTrigger>
              <SelectContent>
                {categoryOptions.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option === "all" ? "All categories" : option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setSearchQuery("");
                setCategoryFilter("all");
              }}
            >
              Clear filters
            </Button>
          </div>

          {filteredCategories.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
              No toolchains match your filters.
            </div>
          ) : (
            filteredCategories.map((category) => (
            <div key={category.category} className="rounded-lg border border-border p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">{category.category}</h2>
                <span className="text-xs text-muted-foreground">
                  {category.entries.length} toolchains
                </span>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {category.entries.map((toolchain) => (
                  <div
                    key={toolchain.id}
                    className="rounded-md border border-border bg-background px-4 py-3"
                  >
                    <div className="text-sm font-medium">{toolchain.title}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Category: {toolchain.category}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button asChild size="sm" variant="outline">
                        <Link
                          href={`/w/${params.workspace}/runs/new?toolchain=${toolchain.id}`}
                        >
                          Run
                        </Link>
                      </Button>
                      <Button asChild size="sm" variant="outline">
                        <Link
                          href={`/w/${params.workspace}/toolchains/${toolchain.id}`}
                        >
                          Open
                        </Link>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
