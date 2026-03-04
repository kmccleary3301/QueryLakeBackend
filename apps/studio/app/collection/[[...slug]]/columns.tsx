"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { LucideLoader2, Minus, Trash } from "lucide-react";
// import type { ColumnSchema } from "./schema";
// import ColumnSchema
import { format, set } from "date-fns";
import { getStatusColor } from "@/lib/request/status-code";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { DataTableColumnHeader } from "@/components/data-table/data-table-column-header";
import TextWithTooltip from "@/components/custom/text-with-tooltip";
import { UTCDate } from "@date-fns/utc";
import { z } from "zod";
// import { createParser, createSerializer, createSearchParamsCache, parseAsArrayOf, parseAsBoolean, parseAsInteger, parseAsString, parseAsTimestamp, useQueryStates } from "nuqs";
import {
  createParser,
  createSearchParamsCache,
  createSerializer,
  parseAsArrayOf,
  parseAsBoolean,
  parseAsInteger,
  parseAsString,
  parseAsTimestamp,
  type inferParserType,
} from "nuqs/server";
import { ARRAY_DELIMITER, RANGE_DELIMITER, SLIDER_DELIMITER, SORT_DELIMITER } from "@/lib/delimiters";

import { Percentile } from "@/lib/request/percentile";
import { Button } from "@/components/ui/button";

export type MakeArray<T> = {
  [P in keyof T]: T[P][];
};

const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

export const REGIONS = ["ams", "fra", "gru", "hkg", "iad", "syd"] as const;

export const regions: Record<string, string> = {
  ams: "Amsterdam",
  fra: "Frankfurt",
  gru: "Sao Paulo",
  hkg: "Hong Kong",
  iad: "Washington D.C.",
  syd: "Sydney",
};


export type InfiniteQueryMeta = {
  totalRowCount: number;
  filterRowCount: number;
  totalFilters: MakeArray<ColumnSchema>;
  currentPercentiles: Record<Percentile, number>;
  chartData: { timestamp: number; [key: string]: number }[];
};



export interface CollectionDocument {
  id: string;
  file_name: string;
  creation_timestamp: number;
  integrity_sha256: string;
  size_bytes: number;
  encryption_key_secure: string;
  organization_document_collection_hash_id: string | null;
  user_document_collection_hash_id: string;
  global_document_collection_hash_id: string | null;
  toolchain_session_id: string | null;
  website_url: string | null;
  blob_id: string;
  blob_dir: string;
  finished_processing: 0 | 1 | 2 | 3 | 4;
  md: Record<string, any>;
  bm25_score: number;
}


export const columnSchema = z.object({
  id: z.string(),
  file_name: z.string(),
  creation_timestamp: z.number(),
  integrity_sha256: z.string(),
  size_bytes: z.number(),
  document_collection_id: z.string(),
  toolchain_session_id: z.string().nullable(),
  website_url: z.string().nullable(),
  blob_id: z.string().nullable(),
  blob_dir: z.string().nullable(),
  finished_processing: z.number(),
  md: z.record(z.any()),
  bm25_score: z.number().nullable()
});

export const parseAsSort = createParser({
  parse(queryValue) {
    const [id, desc] = queryValue.split(SORT_DELIMITER);
    if (!id && !desc) return null;
    return { id, desc: desc === "desc" };
  },
  serialize(value) {
    return `${value.id}.${value.desc ? "desc" : "asc"}`;
  },
});


export const searchParamsParser = {
  // CUSTOM FILTERS
  auth: parseAsString,
  collection_id: parseAsString,
  file_name: parseAsString,
  creation_timestamp: parseAsArrayOf(parseAsTimestamp, RANGE_DELIMITER),
  size_bytes: parseAsArrayOf(parseAsInteger, SLIDER_DELIMITER),
  finished_processing: parseAsArrayOf(parseAsBoolean, ARRAY_DELIMITER),
  
  // REQUIRED FOR SORTING & PAGINATION
  sort: parseAsSort,
  size: parseAsInteger.withDefault(10),
  start: parseAsInteger.withDefault(0),
  
  // REQUIRED FOR SELECTION
  id: parseAsString,
};

export const searchParamsCache = createSearchParamsCache(searchParamsParser);
export const searchParamsSerializer = createSerializer(searchParamsParser);
export type SearchParamsType = inferParserType<typeof searchParamsParser>;

const processingDisplay = {
  0: () => (
    <p className="font-mono text-xs text-primary/50">N/A</p>
  ),
  1: () => (
    <div>
      <div className="h-4 w-[90px] bg-accent flex items-center space-x-1 rounded-full px-2">
        <LucideLoader2 className="w-3 h-3 text-primary animate-spin" />
        <p className="text-xs">Processing</p>
      </div>
    </div>
  ),
  2: () => (
    <p className="text-xs text-primary/50 bg-destructive">Failed</p>
  ),
  3: () => (
    <p className="text-green-500">Done</p>
  ),
  4: () => (
    <p className="text-purple-500">Done</p>
  ),
}

function file_size_as_string(bytes: number, return_i : boolean): string {
	const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
	if (bytes === 0) return '0 B';
	const i = Math.floor(Math.log(bytes) / Math.log(1024));
  if (return_i)
    return i.toString()
	return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export type ColumnSchema = z.infer<typeof columnSchema>;

export const columns: ColumnDef<ColumnSchema>[] = [
  {
    id: "id",
    accessorKey: "id",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="ID" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("id") as string;
      return <TextWithTooltip className="font-mono max-w-[85px]" text={value} />
    },
    enableHiding: true,
    enableSorting: false, // Sorting isn't consistent with the backend on id.
  },
  {
    id: "integrity_sha256",
    accessorKey: "integrity_sha256",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Integrity" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("integrity_sha256") as string;
      return <TextWithTooltip className="font-mono max-w-[85px]" text={value} />
    },
    enableHiding: true,
    enableSorting: true,
    sortingFn: "text"
  },
  {
    id: "file_name",
    accessorKey: "file_name",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Name" />
    ),
    enableSorting: true, // Enable sorting for this column
    sortingFn: "text",
    cell: ({ row }) => {
      const value = row.getValue("file_name") as string;
      return (
        <HoverCard openDelay={0} closeDelay={0}>
          <HoverCardTrigger asChild>
            <div className="max-w-[300px] italic">
              {value}
            </div>
          </HoverCardTrigger>
          <HoverCardContent
            side="right"
            align="center"
            alignOffset={-4}
            className="p-2 w-auto z-10"
          >
            <p>{value}</p>
          </HoverCardContent>
        </HoverCard>
      )
    },
    meta: {
      label: "Name",
    },
  },
  {
    accessorKey: "creation_timestamp",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Created" />
    ),
    enableSorting: true,
    sortingFn: "basic",
    cell: ({ row }) => {
      const date = new Date((row.getValue("creation_timestamp") as number)*1000);
      return (
        <HoverCard openDelay={0} closeDelay={0}>
          <HoverCardTrigger asChild>
            <div className="font-mono whitespace-nowrap">
              {format(date, "LLL dd, y HH:mm:ss")}
            </div>
          </HoverCardTrigger>
          <HoverCardContent
            side="left"
            align="center"
            alignOffset={-4}
            className="p-2 w-auto z-10"
          >
            <dl className="flex flex-col gap-1">
              <div className="flex gap-4 text-sm justify-between items-center">
                <dt className="text-muted-foreground">Timestamp</dt>
                <dd className="font-mono truncate">{date.getTime()}</dd>
              </div>
              <div className="flex gap-4 text-sm justify-between items-center">
                <dt className="text-muted-foreground">UTC</dt>
                <dd className="font-mono truncate">
                  {format(new UTCDate(date), "LLL dd, y HH:mm:ss")}
                </dd>
              </div>
              <div className="flex gap-4 text-sm justify-between items-center">
                <dt className="text-muted-foreground">{timezone}</dt>
                <dd className="font-mono truncate">
                  {format(date, "LLL dd, y HH:mm:ss")}
                </dd>
              </div>
            </dl>
          </HoverCardContent>
        </HoverCard>
      );
    },
    filterFn: "inDateRange",
    meta: {
      // headerClassName: "w-[182px]",
    },
  },
  {
    accessorKey: "size_bytes",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Size" />
    ),
    enableSorting: true,
    sortingFn: "basic",
    cell: ({ row }) => {
      const value = row.getValue("size_bytes");
      if (typeof value === "undefined") {
        return <Minus className="h-4 w-4 text-muted-foreground/50" />;
      }
      if (typeof value === "number") {
        const file_size_string = file_size_as_string(value, false);
        const order_string = file_size_as_string(value, true);
        const colors = getStatusColor(parseInt(order_string));
        return <div className={`${colors.text} font-mono`}>{file_size_string}</div>;
      }
      return <div className="text-muted-foreground">{`${value}`}</div>;
    }
  },
  {
    // TODO: make it a type of MethodSchema!
    accessorKey: "finished_processing",
    enableSorting: true,
    sortingFn: "basic",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Processing" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("finished_processing") as 0 | 1 | 2 | 3 | 4;
      return processingDisplay[value]();
    }
  }
];

export const columnFilterSchema = z.object({
  id: z.string().optional(),
  file_name: z.string().optional(),
  creation_timestamp: z
    .string()
    .transform((val) => val.split(RANGE_DELIMITER).map(Number))
    .pipe(z.coerce.date().array())
    .optional(),
  integrity_sha256: z.string().optional(),
  size_bytes: z
    .string()
    .transform((val) => val.split(SLIDER_DELIMITER))
    .pipe(z.coerce.number().array().max(2))
    .optional(),
  finished_processing: z
    .string()
    .transform((val) => val.split(ARRAY_DELIMITER))
    .pipe(z.coerce.number().array())
    .optional(),
  website_url: z.string().optional(),
  toolchain_session_id: z.string().optional(),
  blob_id: z.string().optional(),
  blob_dir: z.string().optional(),
});
