"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { useContextAction } from "@/app/context-provider";
import { openDocument, QueryLakeFetchDocument } from "@/hooks/querylakeAPI";
import type { fetch_document_result } from "@/hooks/querylakeAPI";

type ChunkRow = {
  id: string | number | Array<string> | Array<number>;
  document_chunk_number?: number | [number, number] | null;
  text: string;
  bm25_score?: number | null;
  md?: Record<string, unknown>;
};

const DEFAULT_PAGE_SIZE = 50;

const chunkNumberToString = (
  chunkNumber?: number | [number, number] | null
) => {
  if (chunkNumber == null) return "—";
  if (Array.isArray(chunkNumber)) return `${chunkNumber[0]}–${chunkNumber[1]}`;
  return String(chunkNumber);
};

const makeChunkQuery = (documentId: string, queryText: string) => {
  const filter = `document_id:\"${documentId}\"`;
  const trimmed = queryText.trim();
  if (!trimmed) return filter;
  return `${filter} ${trimmed}`;
};

export default function DocumentPage() {
  const params = useParams<{ workspace: string; documentId: string }>()!;
  const searchParams = useSearchParams();
  const { userData, authReviewed, loginValid } = useContextAction();

  const [document, setDocument] = useState<fetch_document_result | null>(null);
  const [queryText, setQueryText] = useState(() => searchParams?.get("q") ?? "");
  const [chunks, setChunks] = useState<ChunkRow[]>([]);
  const [offset, setOffset] = useState(0);
  const [loadingDoc, setLoadingDoc] = useState(true);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [copiedChunkId, setCopiedChunkId] = useState<string | null>(null);

  const collectionId = document?.collection_id ?? null;

  const fetchDoc = useCallback(() => {
    if (!userData?.auth) return;
    setLoadingDoc(true);
    QueryLakeFetchDocument({
      auth: userData.auth,
      document_id: params.documentId,
      onFinish: (result) => {
        if (result === false) {
          setDocument(null);
        } else {
          setDocument(result);
        }
        setLoadingDoc(false);
      },
    });
  }, [params.documentId, userData?.auth]);

  const fetchChunks = useCallback(
    async ({
      nextOffset,
      append,
    }: {
      nextOffset: number;
      append: boolean;
    }) => {
      if (!userData?.auth || !collectionId) return;
      setLoadingChunks(true);
      try {
        const response = await fetch("/api/search_bm25", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            auth: userData.auth,
            collection_ids: [collectionId],
            table: "document_chunk",
            group_chunks: false,
            query: makeChunkQuery(params.documentId, queryText),
            sort_by: "document_chunk_number",
            sort_dir: "ASC",
            limit: DEFAULT_PAGE_SIZE,
            offset: nextOffset,
          }),
        });
        const payload = (await response.json()) as {
          success: boolean;
          result?: ChunkRow[];
        };
        if (!payload.success) {
          setHasMore(false);
          if (!append) setChunks([]);
          return;
        }
        const nextRows = payload.result ?? [];
        setChunks((prev) => (append ? [...prev, ...nextRows] : nextRows));
        setHasMore(nextRows.length === DEFAULT_PAGE_SIZE);
        setOffset(nextOffset);
      } finally {
        setLoadingChunks(false);
      }
    },
    [collectionId, params.documentId, queryText, userData?.auth]
  );

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) {
      setLoadingDoc(false);
      setDocument(null);
      return;
    }
    fetchDoc();
  }, [authReviewed, loginValid, userData?.auth, fetchDoc]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (!collectionId) return;
    fetchChunks({ nextOffset: 0, append: false });
  }, [authReviewed, loginValid, userData?.auth, collectionId, fetchChunks]);

  const headerTitle = useMemo(() => {
    if (document?.file_name) return document.file_name;
    return params.documentId;
  }, [document?.file_name, params.documentId]);

  if (!authReviewed) {
    return (
      <div className="space-y-6">
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-5 w-56" />
          <Skeleton className="h-4 w-72" />
          <Skeleton className="h-4 w-64" />
        </div>
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Document</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to view documents.
        </p>
        <Button asChild size="sm">
          <Link href="/auth/login">Go to login</Link>
        </Button>
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
              <BreadcrumbSeparator />
              {collectionId ? (
                <>
                  <BreadcrumbItem>
                    <BreadcrumbLink href={`/w/${params.workspace}/collections`}>
                      Collections
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator />
                  <BreadcrumbItem>
                    <BreadcrumbLink
                      href={`/w/${params.workspace}/collections/${collectionId}`}
                    >
                      Collection
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator />
                </>
              ) : null}
              <BreadcrumbItem>
                <BreadcrumbPage>Document</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">{headerTitle}</h1>
          <p className="text-sm text-muted-foreground">
            Document id: {params.documentId}
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {collectionId ? (
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/collections/${collectionId}`}>
                Back to collection
              </Link>
            </Button>
          ) : null}
          {collectionId && queryText.trim() ? (
            <Button asChild size="sm" variant="outline">
              <Link
                href={`/w/${params.workspace}/collections/${collectionId}?q=${encodeURIComponent(
                  queryText
                )}`}
              >
                Back to search
              </Link>
            </Button>
          ) : null}
          <Button
            size="sm"
            variant="outline"
            disabled={!document || loadingDoc}
            onClick={() => {
              if (!userData?.auth) return;
              openDocument({ auth: userData.auth, document_id: params.documentId });
            }}
          >
            Download
          </Button>
          <Button size="sm" variant="outline" onClick={fetchDoc} disabled={loadingDoc}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div className="rounded-lg border border-border p-5 text-sm">
          <div className="font-semibold">Metadata</div>
          {loadingDoc ? (
            <div className="mt-3 space-y-2">
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-1/3" />
            </div>
          ) : document ? (
            <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
              <div className="flex justify-between gap-4">
                <span>File</span>
                <span className="truncate font-medium text-foreground">
                  {document.file_name}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Created</span>
                <span className="font-medium text-foreground">
                  {new Date(document.creation_timestamp * 1000).toISOString()}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Size</span>
                <span className="font-medium text-foreground">
                  {document.size_bytes.toLocaleString()} bytes
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Status</span>
                <span className="font-medium text-foreground">
                  {document.finished_processing ? "Ready" : "Processing"}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Collection</span>
                <span className="font-medium text-foreground">
                  {document.collection_id}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Chunks</span>
                <span className="font-medium text-foreground">
                  {document.chunk_count}
                </span>
              </div>
            </div>
          ) : (
            <div className="mt-3 text-sm text-muted-foreground">
              Document not found or unavailable.
            </div>
          )}
        </div>

        <div className="rounded-lg border border-border p-5 text-sm">
          <div className="font-semibold">Search chunks</div>
          <p className="mt-1 text-xs text-muted-foreground">
            Searches within this document’s chunks.
          </p>
          <div className="mt-4 flex gap-2">
            <Input
              value={queryText}
              placeholder="Search text"
              onChange={(event) => setQueryText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  fetchChunks({ nextOffset: 0, append: false });
                }
              }}
            />
            <Button
              variant="outline"
              disabled={!collectionId || loadingChunks}
              onClick={() => fetchChunks({ nextOffset: 0, append: false })}
            >
              Search
            </Button>
          </div>
          <div className="mt-3 text-xs text-muted-foreground">
            Tip: field filters are supported (e.g. `document_chunk_number:1`).
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[160px]">Chunk</TableHead>
              <TableHead>Text</TableHead>
              <TableHead className="w-[120px]">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loadingDoc ? (
              <TableRow>
                <TableCell colSpan={3} className="py-6 text-center text-sm text-muted-foreground">
                  Loading document...
                </TableCell>
              </TableRow>
            ) : !document ? (
              <TableRow>
                <TableCell colSpan={3} className="py-6 text-center text-sm text-muted-foreground">
                  Document unavailable.
                </TableCell>
              </TableRow>
            ) : loadingChunks ? (
              <TableRow>
                <TableCell colSpan={3} className="py-6 text-center text-sm text-muted-foreground">
                  Loading chunks...
                </TableCell>
              </TableRow>
            ) : chunks.length === 0 ? (
              <TableRow>
                <TableCell colSpan={3} className="py-6 text-center text-sm text-muted-foreground">
                  No chunks found.
                </TableCell>
              </TableRow>
            ) : (
              chunks.map((chunk) => (
                <TableRow key={String(chunk.id)}>
                  <TableCell className="text-xs text-muted-foreground">
                    {chunkNumberToString(chunk.document_chunk_number)}
                  </TableCell>
                  <TableCell className="text-sm">
                    <div className="line-clamp-3 whitespace-pre-wrap">
                      {chunk.text}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        navigator.clipboard.writeText(chunk.text);
                        setCopiedChunkId(String(chunk.id));
                        window.setTimeout(() => setCopiedChunkId(null), 1500);
                      }}
                    >
                      {copiedChunkId === String(chunk.id) ? "Copied" : "Copy"}
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {hasMore ? (
        <div className="flex justify-center">
          <Button
            variant="outline"
            disabled={loadingChunks}
            onClick={() =>
              fetchChunks({ nextOffset: offset + DEFAULT_PAGE_SIZE, append: true })
            }
          >
            Load more
          </Button>
        </div>
      ) : null}
    </div>
  );
}
