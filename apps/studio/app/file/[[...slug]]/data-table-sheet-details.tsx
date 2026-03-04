"use client";

import { ChevronDown, ChevronUp, Download, X } from "lucide-react";
import * as React from "react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { Table } from "@tanstack/react-table";

// import { ColumnSchema } from "./schema";
import { ColumnSchema } from "./columns";
import { Check } from "lucide-react";
import CopyToClipboardContainer from "@/components/custom/copy-to-clipboard-container";
import { cn } from "@/lib/utils";
import {
  formatDate,
} from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { getStatusColor } from "@/lib/request/status-code";
import { Skeleton } from "@/components/ui/skeleton";
import { Percentile } from "@/lib/request/percentile";
import { SheetFooter } from "@/components/ui/sheet";
import MarkdownCodeBlock from "@/components/markdown/markdown-code-block";

export interface DataTableSheetDetailsProps<TData> {
  table: Table<TData>;
  title?: string;
  titleClassName?: string;
  children?: React.ReactNode;
  onDelete?: () => void;
  onDownload?: () => void;
}

export function DocumentChunkTableSheetDetails<TData>({
  table,
  title,
  titleClassName,
  children,
  onDelete,
  onDownload,
}: DataTableSheetDetailsProps<TData>) {
  const selectedRowKey =
    Object.keys(table.getState().rowSelection)?.[0] || undefined;

  const index = table
    .getCoreRowModel()
    .flatRows.findIndex((row) => row.id === selectedRowKey);

  const nextId = React.useMemo(
    () => table.getCoreRowModel().flatRows[index + 1]?.id,
    [index, table]
  );

  const prevId = React.useMemo(
    () => table.getCoreRowModel().flatRows[index - 1]?.id,
    [index, table]
  );

  const onPrev = React.useCallback(() => {
    if (prevId) table.setRowSelection({ [prevId]: true });
  }, [prevId, table]);

  const onNext = React.useCallback(() => {
    if (nextId) table.setRowSelection({ [nextId]: true });
  }, [nextId, table]);

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (!selectedRowKey) return;

      if (e.key === "ArrowUp") {
        e.preventDefault();
        onPrev();
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        onNext();
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, [selectedRowKey, onNext, onPrev]);

  return (
    <Sheet
      open={!!selectedRowKey}
      onOpenChange={() => table.toggleAllRowsSelected(false)}
    >
      <SheetContent className="sm:max-w-lg overflow-y-auto p-0">
        <SheetHeader className="sticky top-0 border-b bg-background p-4">
          <div className="flex items-center justify-between gap-2">
            <SheetTitle className={cn(titleClassName, "text-left truncate")}>
              {title}
            </SheetTitle>
          </div>
        </SheetHeader>
        <SheetDescription className="sr-only">
          Selected row details
        </SheetDescription>
        <div className="p-4">{children}</div>
        <SheetFooter className="px-4">
          <Button variant="default" onClick={onDownload} className="p-2 h-8">
          <span className="">Open Document</span>
          </Button>
          <SheetClose asChild>
            <Button variant="destructive" type="submit" onSubmit={onDelete} className="h-8">
              <span className="">Delete</span>
            </Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}


interface SheetDetailsContentProps
  extends React.HTMLAttributes<HTMLDListElement> {
  data?: ColumnSchema;
  percentiles?: Record<Percentile, number>;
  filterRows: number;
}

export function DocumentChunkSheetDetailsContent({
  data,
  filterRows,
  ...props
}: SheetDetailsContentProps) {
  const [open, setOpen] = React.useState(false);

  if (!data) return <CollectionSheetDetailsContentSkeleton />;

  const statusColor = getStatusColor(200);
  

  return (
    <dl {...props}>
      <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">Name</dt>
        <dd className="font-mono">
          <span className="text-muted-foreground text-wrap">{data.document_name}</span>
        </dd>
      </div>
      <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">ID</dt>
        <dd className="font-mono truncate">{data.id}</dd>
      </div>
      <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">Chunk Index</dt>
        <dd className="font-mono truncate">{data.document_chunk_number}</dd>
      </div>
      {/* <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">Processing</dt>
        <dd>
          {data.finished_processing ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <X className="h-4 w-4 text-red-500" />
          )}
        </dd>
      </div> */}
      <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">Upload Date</dt>
        <dd className="font-mono text-right">{formatDate(new Date(1000*(data.creation_timestamp)))}</dd>
      </div>
      <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
        <dt className="text-muted-foreground">File Type</dt>
        <dd>
          <Badge
            variant="outline"
            className={`${statusColor.bg} ${statusColor.border} ${statusColor.text} font-mono`}
          >
            {data.document_name.split(".").pop()?.toLowerCase()}
          </Badge>
        </dd>
      </div>
      {(data.document_md.integrity_sha256 as string) && (
        <div className="flex gap-4 py-2 border-b text-sm justify-between items-center">
          <dt className="text-muted-foreground">Integrity</dt>
          <dd className="font-mono truncate text-xs">
            {data.document_md.integrity_sha256.match(/.{32}/g)?.map((str : string, i : number) => (
              <div key={i} className="w-full">
                {str}
              </div>
            ))}
          </dd>
        </div>
      )}
      <div className="flex flex-col gap-2 py-2 border-b text-sm text-left">
        <dt className="text-muted-foreground">Chunk Metadata</dt>
        {/* <CopyToClipboardContainer className="rounded-md bg-destructive/30 border border-destructive/50 p-2 whitespace-pre-wrap break-all font-mono">
          {JSON.stringify(data.md, null, 2)}
        </CopyToClipboardContainer> */}
        <MarkdownCodeBlock lang="json" text={JSON.stringify(data.md, null, 2)}/>
      </div>
      <div className="flex flex-col gap-2 py-2 border-b text-sm text-left">
        <dt className="text-muted-foreground">Document Metadata</dt>
        {/* <CopyToClipboardContainer className="rounded-md bg-destructive/30 border border-destructive/50 p-2 whitespace-pre-wrap break-all font-mono">
          {JSON.stringify(data.md, null, 2)}
        </CopyToClipboardContainer> */}
        <MarkdownCodeBlock lang="json" text={JSON.stringify(data.document_md, null, 2)}/>
      </div>
      
      {/* {data.message ? (
        <div className="flex flex-col gap-2 py-2 border-b text-sm text-left">
          <dt className="text-muted-foreground">Message</dt>
          <CopyToClipboardContainer className="rounded-md bg-destructive/30 border border-destructive/50 p-2 whitespace-pre-wrap break-all font-mono">
            {JSON.stringify(data.message, null, 2)}
          </CopyToClipboardContainer>
        </div>
      ) : null} */}
      {/* <div className="flex flex-col gap-2 py-2 text-sm text-left">
        <dt className="text-muted-foreground">Headers</dt>
        <CopyToClipboardContainer className="rounded-md bg-muted/50 border p-2 whitespace-pre-wrap break-all font-mono">
          {JSON.stringify(data.headers, null, 2)}
        </CopyToClipboardContainer>
      </div> */}
    </dl>
  );
}

const skeleton = {
  ID: "h-5 w-52",
  Success: "h-5 w-5",
  Date: "h-5 w-36",
  "Status Code": "h-5 w-12",
  Host: "h-5 w-24",
  Pathname: "h-5 w-56",
  Region: "h-5 w-12",
  Latency: "h-5 w-16",
  Percentile: "h-5 w-12",
  "Timing Phases": "h-5 w-52",
  Message: "h-5 w-52",
};

export function CollectionSheetDetailsContentSkeleton() {
  return (
    <dl>
      {Object.entries(skeleton).map(([key, size]) => (
        <div
          key={key}
          className="flex gap-4 py-2 border-b text-sm justify-between items-center"
        >
          <dt className="text-muted-foreground">{key}</dt>
          <dd>
            <Skeleton className={size} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
