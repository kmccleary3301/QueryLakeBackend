"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import Link from "next/link";
import FileDropzone from "@/components/ui/file-dropzone";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
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
  fetch_collection_document_type,
  fetchCollection,
  fetchCollectionDocuments,
} from "@/hooks/querylakeAPI";
import uploadFiles from "@/hooks/upload-files";

type CollectionSummary = {
  title: string;
  description: string;
  type: "user" | "organization" | "global";
  owner: string;
  public: boolean;
  document_count: number;
};

type CollectionSearchRow = {
  id: string | number | Array<string> | Array<number>;
  document_id?: string | null;
  document_name: string;
  document_chunk_number?: number | [number, number] | null;
  bm25_score?: number | null;
  text: string;
};

const chunkNumberToString = (
  chunkNumber?: number | [number, number] | null
) => {
  if (chunkNumber == null) return "—";
  if (Array.isArray(chunkNumber)) return `${chunkNumber[0]}–${chunkNumber[1]}`;
  return String(chunkNumber);
};

const buildDocumentSearchQuery = (
  queryText: string,
  chunkNumber?: number | [number, number] | null
) => {
  const trimmed = queryText.trim();
  if (!trimmed) return "";
  if (chunkNumber == null) return trimmed;
  const firstChunk = Array.isArray(chunkNumber) ? chunkNumber[0] : chunkNumber;
  if (typeof firstChunk !== "number") return trimmed;
  return `document_chunk_number:${firstChunk} ${trimmed}`.trim();
};

const SEARCH_PAGE_SIZE = 20;

export default function CollectionPage() {
  const params = useParams<{ workspace: string; collectionId: string }>()!;
  const searchParams = useSearchParams();
  const { userData, authReviewed, loginValid } = useContextAction();
  const [collection, setCollection] = useState<CollectionSummary | null>(null);
  const [documents, setDocuments] = useState<fetch_collection_document_type[]>(
    []
  );
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState<
    { name: string; progress: number }[]
  >([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const dropzoneRef = useRef<HTMLDivElement | null>(null);
  const initialSearchQuery = useMemo(() => searchParams?.get("q") ?? "", [searchParams]);
  const [searchQuery, setSearchQuery] = useState(initialSearchQuery);
  const [searchResults, setSearchResults] = useState<CollectionSearchRow[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchOffset, setSearchOffset] = useState(0);
  const [searchHasMore, setSearchHasMore] = useState(false);
  const autoSearchRanRef = useRef(false);

  const refreshDocuments = useCallback(() => {
    if (!userData?.auth) return;
    fetchCollectionDocuments({
      auth: userData.auth,
      collection_id: params.collectionId,
      onFinish: (result) => {
        setDocuments(result ?? []);
      },
    });
  }, [params.collectionId, userData?.auth]);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) {
      setLoading(false);
      return;
    }
    setLoading(true);
    fetchCollection({
      auth: userData.auth,
      collection_id: params.collectionId,
      onFinish: (result) => {
        if (result) {
          setCollection(result as CollectionSummary);
        }
        setLoading(false);
      },
    });
    refreshDocuments();
  }, [authReviewed, loginValid, userData?.auth, params.collectionId, refreshDocuments]);

  useEffect(() => {
    if (!userData?.auth) return;
    if (!documents.some((doc) => !doc.finished_processing)) return;
    const interval = setInterval(() => {
      refreshDocuments();
      fetchCollection({
        auth: userData.auth,
        collection_id: params.collectionId,
        onFinish: (result) => {
          if (result) {
            setCollection(result as CollectionSummary);
          }
        },
      });
    }, 8000);
    return () => clearInterval(interval);
  }, [documents, userData?.auth, params.collectionId, refreshDocuments]);

  const runSearch = useCallback(
    async ({
      nextOffset,
      append,
    }: {
      nextOffset: number;
      append: boolean;
    }) => {
      if (!userData?.auth) return;
      const trimmed = searchQuery.trim();
      if (!trimmed) {
        setSearchResults([]);
        setSearchHasMore(false);
        setSearchOffset(0);
        return;
      }

      setSearchLoading(true);
      try {
        const response = await fetch("/api/search_bm25", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auth: userData.auth,
            collection_ids: [params.collectionId],
            table: "document_chunk",
            group_chunks: true,
            query: trimmed,
            limit: SEARCH_PAGE_SIZE,
            offset: nextOffset,
          }),
        });
        const payload = (await response.json()) as {
          success: boolean;
          result?: CollectionSearchRow[];
        };
        if (!payload.success) {
          setSearchHasMore(false);
          if (!append) setSearchResults([]);
          return;
        }
        const nextRows = payload.result ?? [];
        setSearchResults((prev) => (append ? [...prev, ...nextRows] : nextRows));
        setSearchHasMore(nextRows.length === SEARCH_PAGE_SIZE);
        setSearchOffset(nextOffset);
      } finally {
        setSearchLoading(false);
      }
    },
    [params.collectionId, searchQuery, userData?.auth]
  );

  useEffect(() => {
    if (autoSearchRanRef.current) return;
    if (!initialSearchQuery.trim()) return;
    if (!authReviewed || !loginValid || !userData?.auth) return;
    autoSearchRanRef.current = true;
    runSearch({ nextOffset: 0, append: false });
  }, [authReviewed, initialSearchQuery, loginValid, runSearch, userData?.auth]);

  const startUpload = async (files: File[]) => {
    if (!userData?.auth || files.length === 0) return;
    setUploadError(null);
    setUploading(files.map((file) => ({ name: file.name, progress: 0 })));

    const responses = await uploadFiles({
      files,
      url: "/upload/",
      parameters: {
        auth: userData.auth,
        collection_hash_id: params.collectionId,
      },
      on_upload_progress: (progress, index) => {
        setUploading((prev) =>
          prev.map((entry, idx) =>
            idx === index ? { ...entry, progress } : entry
          )
        );
      },
      on_response: (response) => {
        if ((response as { success?: boolean }).success === false) {
          setUploadError("One or more files failed to upload.");
        }
      },
    });

    setUploading([]);
    refreshDocuments();
    fetchCollection({
      auth: userData.auth,
      collection_id: params.collectionId,
      onFinish: (result) => {
        if (result) {
          setCollection(result as CollectionSummary);
        }
      },
    });
    const failures = responses.filter(
      (response) => (response as { success?: boolean }).success === false
    );
    if (failures.length > 0) {
      toast(`Uploaded with ${failures.length} failure(s).`);
    } else {
      toast("Upload complete.");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
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
                <BreadcrumbPage>{params.collectionId}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">
            {collection?.title ?? `Collection: ${params.collectionId}`}
          </h1>
          <p className="text-sm text-muted-foreground">
            Workspace {params.workspace}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline">
            <Link href={`/w/${params.workspace}/collections`}>
              Back to collections
            </Link>
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              dropzoneRef.current?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            Upload documents
          </Button>
          <Button asChild variant="outline">
            <Link href={`/w/${params.workspace}/files`}>
              Open in Files
            </Link>
          </Button>
        </div>
      </div>

      <div ref={dropzoneRef} className="rounded-lg border border-border p-4">
        <div className="text-sm font-medium">Upload documents</div>
        <p className="mt-1 text-xs text-muted-foreground">
          Drop files here to ingest them into this collection.
        </p>
        {!loading && documents.some((doc) => !doc.finished_processing) ? (
          <p className="mt-2 text-xs text-muted-foreground">
            Processing is still running for{" "}
            {documents.filter((doc) => !doc.finished_processing).length} document(s).
            This page auto-refreshes while ingestion completes.
          </p>
        ) : null}
        <div className="mt-4">
          <FileDropzone
            multiple
            onFile={(files) => {
              startUpload(files);
            }}
          />
        </div>
        {uploadError ? (
          <div className="mt-3 text-xs text-destructive">{uploadError}</div>
        ) : null}
        {uploading.length > 0 && (
          <div className="mt-4 space-y-2 text-xs text-muted-foreground">
            {uploading.map((entry) => (
              <div key={entry.name} className="flex items-center justify-between">
                <span>{entry.name}</span>
                <span>{entry.progress}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-5 w-44" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-52" />
          <Skeleton className="h-4 w-36" />
        </div>
      ) : !collection ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Collection not found or unavailable.
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <div className="rounded-lg border border-border p-5 space-y-3 text-sm">
            <div className="font-semibold">Overview</div>
            <div className="text-muted-foreground">
              {collection.description || "No description provided."}
            </div>
            <div className="grid gap-2 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Type</span>
                <span className="font-medium text-foreground">
                  {collection.type}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Owner</span>
                <span className="font-medium text-foreground">
                  {collection.owner}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Documents</span>
                <span className="font-medium text-foreground">
                  {collection.document_count}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Visibility</span>
                <span className="font-medium text-foreground">
                  {collection.public ? "Public" : "Private"}
                </span>
              </div>
            </div>
          </div>
          <div className="rounded-lg border border-border p-5 text-sm">
            <div className="font-semibold">Next steps</div>
            <ul className="mt-3 space-y-2 text-muted-foreground">
              <li>Upload files to populate this collection.</li>
              <li>Review parsing status and metadata.</li>
              <li>Run retrieval queries once indexed.</li>
            </ul>
          </div>
        </div>
      )}

      <div className="rounded-lg border border-border p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-sm font-semibold">Search in collection</div>
            <p className="mt-1 text-xs text-muted-foreground">
              BM25 search across indexed document chunks in this collection.
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-2 md:flex-row md:items-center">
          <Input
            value={searchQuery}
            placeholder='Search chunks (e.g. title:"foo" or document_chunk_number:1)'
            onChange={(event) => setSearchQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                runSearch({ nextOffset: 0, append: false });
              }
            }}
          />
          <Button
            variant="outline"
            disabled={!userData?.auth || searchLoading}
            onClick={() => runSearch({ nextOffset: 0, append: false })}
          >
            Search
          </Button>
        </div>

        <div className="rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[240px]">Document</TableHead>
                <TableHead className="w-[140px]">Chunk</TableHead>
                <TableHead>Text</TableHead>
                <TableHead className="w-[120px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {searchLoading ? (
                <TableRow>
                  <TableCell
                    colSpan={4}
                    className="py-6 text-center text-sm text-muted-foreground"
                  >
                    Searching...
                  </TableCell>
                </TableRow>
              ) : searchResults.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={4}
                    className="py-6 text-center text-sm text-muted-foreground"
                  >
                    {searchQuery.trim()
                      ? "No matches yet."
                      : "Enter a query to search within this collection."}
                  </TableCell>
                </TableRow>
              ) : (
                searchResults.map((row) => {
                  const queryParam = buildDocumentSearchQuery(
                    searchQuery,
                    row.document_chunk_number
                  );
                  const documentHref = row.document_id
                    ? `/w/${params.workspace}/documents/${row.document_id}${
                        queryParam ? `?q=${encodeURIComponent(queryParam)}` : ""
                      }`
                    : null;

                  return (
                    <TableRow key={String(row.id)}>
                      <TableCell className="font-medium">
                        {documentHref ? (
                          <Link href={documentHref} className="hover:underline">
                            {row.document_name}
                          </Link>
                        ) : (
                          row.document_name
                        )}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {chunkNumberToString(row.document_chunk_number)}
                      </TableCell>
                      <TableCell className="text-sm">
                        <div className="line-clamp-3 whitespace-pre-wrap">
                          {row.text}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        {documentHref ? (
                          <Button asChild size="sm" variant="outline">
                            <Link href={documentHref}>Open</Link>
                          </Button>
                        ) : null}
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>

        {searchHasMore ? (
          <div className="flex justify-center">
            <Button
              variant="outline"
              disabled={searchLoading}
              onClick={() =>
                runSearch({
                  nextOffset: searchOffset + SEARCH_PAGE_SIZE,
                  append: true,
                })
              }
            >
              Load more
            </Button>
          </div>
        ) : null}
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Document</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Length</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-[120px]" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {documents.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="py-6 text-center text-sm text-muted-foreground">
                  No documents available.
                </TableCell>
              </TableRow>
            ) : (
              documents.map((doc) => (
                <TableRow key={doc.hash_id}>
                  <TableCell className="font-medium">
                    <Link href={`/w/${params.workspace}/documents/${doc.hash_id}`}>
                      {doc.title}
                    </Link>
                  </TableCell>
                  <TableCell>{doc.size}</TableCell>
                  <TableCell>{doc.length}</TableCell>
                  <TableCell>
                    {doc.finished_processing ? "Ready" : "Processing"}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button asChild size="sm" variant="outline">
                      <Link
                        href={`/w/${params.workspace}/documents/${doc.hash_id}`}
                      >
                        Open
                      </Link>
                    </Button>
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
