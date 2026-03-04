"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import FileDropzone from "@/components/ui/file-dropzone";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
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
  fetchCollectionDocuments,
} from "@/hooks/querylakeAPI";
import uploadFiles from "@/hooks/upload-files";
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
  const [selectedCollection, setSelectedCollection] = useState<string>("");
  const [documents, setDocuments] = useState<fetch_collection_document_type[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState<
    { name: string; progress: number }[]
  >([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const dropzoneRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (collectionGroups.length === 0) {
      refreshCollectionGroups();
    }
  }, [authReviewed, loginValid, userData?.auth, collectionGroups.length, refreshCollectionGroups]);

  const availableCollections = useMemo(() => {
    if (collectionGroups.length === 0) return [];
    if (isPersonalWorkspace(params.workspace)) {
      return collectionGroups
        .filter((group) =>
          ["My Collections", "Global Collections"].includes(group.title)
        )
        .flatMap((group) => group.collections);
    }
    const membership = userData?.memberships.find(
      (member) => member.organization_id === params.workspace
    );
    if (!membership) {
      return collectionGroups.flatMap((group) => group.collections);
    }
    const matchingGroups = collectionGroups.filter(
      (group) => group.title === membership.organization_name
    );
    return matchingGroups.length
      ? matchingGroups.flatMap((group) => group.collections)
      : collectionGroups.flatMap((group) => group.collections);
  }, [collectionGroups, params.workspace, userData?.memberships]);

  useEffect(() => {
    if (!selectedCollection && availableCollections.length > 0) {
      setSelectedCollection(availableCollections[0].hash_id);
    }
  }, [availableCollections, selectedCollection]);

  const refreshDocuments = useCallback(() => {
    if (!userData?.auth || !selectedCollection) return;
    setLoading(true);
    fetchCollectionDocuments({
      auth: userData.auth,
      collection_id: selectedCollection,
      onFinish: (result) => {
        setDocuments(result ?? []);
        setLoading(false);
      },
    });
  }, [selectedCollection, userData?.auth]);

  useEffect(() => {
    refreshDocuments();
  }, [selectedCollection, userData?.auth, refreshDocuments]);

  useEffect(() => {
    if (!userData?.auth) return;
    if (!documents.some((doc) => !doc.finished_processing)) return;
    const interval = setInterval(() => {
      refreshDocuments();
    }, 8000);
    return () => clearInterval(interval);
  }, [documents, userData?.auth, selectedCollection, refreshDocuments]);

  const startUpload = async (files: File[]) => {
    if (!userData?.auth || !selectedCollection || files.length === 0) return;
    setUploadError(null);
    setUploading(files.map((file) => ({ name: file.name, progress: 0 })));

    const responses = await uploadFiles({
      files,
      url: "/upload/",
      parameters: {
        auth: userData.auth,
        collection_hash_id: selectedCollection,
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
                <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbItem>
                <BreadcrumbPage>Files</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Files</h1>
          <p className="text-sm text-muted-foreground">
            Upload, parse, and organize file sources for ingestion.
          </p>
        </div>
        <Button
          variant="outline"
          disabled={!selectedCollection}
          onClick={() => {
            dropzoneRef.current?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          Upload files
        </Button>
      </div>

      <div ref={dropzoneRef} className="rounded-lg border border-border p-4">
        <div className="text-sm font-medium">Upload to selected collection</div>
        <p className="mt-1 text-xs text-muted-foreground">
          Drop files here to ingest them into the selected collection.
        </p>
        {documents.some((doc) => !doc.finished_processing) ? (
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
              if (!selectedCollection) {
                setUploadError("Select a collection before uploading.");
                return;
              }
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

      {!authReviewed ? (
        <div className="rounded-lg border border-border p-5 space-y-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-52" />
        </div>
      ) : !loginValid || !userData ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Sign in to view and upload files.
        </div>
      ) : availableCollections.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          <div>No collections available yet. Create a collection to upload files.</div>
          <div className="mt-4">
            <Button asChild size="sm" variant="outline">
              <Link href={`/w/${params.workspace}/collections/new`}>
                Create collection
              </Link>
            </Button>
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <div className="text-sm text-muted-foreground">Collection</div>
        <Select
          value={selectedCollection}
          onValueChange={(value) => setSelectedCollection(value)}
        >
          <SelectTrigger className="w-[320px]">
            <SelectValue placeholder="Choose a collection" />
          </SelectTrigger>
          <SelectContent>
            {availableCollections.map((collection) => (
              <SelectItem key={collection.hash_id} value={collection.hash_id}>
                {collection.title}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selectedCollection ? (
          <Button asChild size="sm" variant="outline">
            <Link href={`/w/${params.workspace}/collections/${selectedCollection}`}>
              Open collection
            </Link>
          </Button>
        ) : (
          <Button size="sm" variant="outline" disabled>
            Open collection
          </Button>
        )}
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
            {loading ? (
              <TableRow>
                <TableCell colSpan={4} className="py-6 text-center text-sm text-muted-foreground">
                  Loading documents...
                </TableCell>
              </TableRow>
            ) : documents.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="py-6 text-center text-sm text-muted-foreground">
                  No documents found for this collection yet.
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
