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
import MarkdownCodeBlock from "@/components/markdown/markdown-code-block";
import { CHAT_RENDERING_STYLE } from "@/components/markdown/configs";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";

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



export interface DocumentChunk {
  id: string;
  creation_timestamp: number;
  collection_type: string;
  document_id: string;
  document_chunk_number: number;
  collection_id: string;
  document_name: string;
  md: Record<string, any>;
  document_md: Record<string, any>;
  text: string;
}

export const columnSchema = z.object({
  id: z.string(),
  creation_timestamp: z.number(),
  collection_type: z.string(),
  document_id: z.string(),
  document_chunk_number: z.number(),
  collection_id: z.string(),
  document_name: z.string(),
  md: z.record(z.any()),
  document_md: z.record(z.any()),
  text: z.string()
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
  document_name: parseAsString,
  creation_timestamp: parseAsArrayOf(parseAsTimestamp, RANGE_DELIMITER),
  collection_type: parseAsArrayOf(parseAsString, ARRAY_DELIMITER),
  
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
    enableSorting: true,
    sortingFn: "alphanumeric"
  },
  {
    id: "document_id",
    accessorKey: "document_id",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Document ID" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("document_id") as string;
      return <TextWithTooltip className="font-mono max-w-[85px]" text={value} />
    },
    enableHiding: true,
    enableSorting: true,
    sortingFn: "alphanumeric"
  },
  {
    id: "document_name",
    accessorKey: "document_name",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Name" />
    ),
    enableSorting: true, // Enable sorting for this column
    cell: ({ row }) => {
      const value = row.getValue("document_name") as string;
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
    accessorKey: "document_chunk_number",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Index" />
    ),
    enableSorting: true,
    sortingFn: "basic",
    cell: ({ row }) => {
      const value = row.getValue("document_chunk_number");
      if (typeof value === "undefined") {
        return <Minus className="h-4 w-4 text-muted-foreground/50" />;
      }
      // if (typeof value === "number") {
      //   const file_size_string = file_size_as_string(value, false);
      //   const order_string = file_size_as_string(value, true);
      //   const colors = getStatusColor(parseInt(order_string));
      //   return <div className={`${colors.text} font-mono`}>{file_size_string}</div>;
      // }
      return <div className="text-muted-foreground">{`${value}`}</div>;
    }
  },
  {
    id: "text",
    accessorKey: "text",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Content" />
    ),
    enableSorting: false, // Enable sorting for this column
    cell: ({ row }) => {
      const value = row.getValue("text") as string;
      return (
        // <div className="pr-8">
        //   {value}
        // </div>
        <div style={{marginLeft: "0.1rem" }}> {/* The left pad doesn't render for some reason */}
          <MarkdownRenderer
            className="w-[70vw] lg:w-[55vw] xl:w-[40vw]"
            input={value} 
            finished={false}
            config={CHAT_RENDERING_STYLE}
          />
        </div>
      )
    },
    meta: {
      label: "Name",
    },
  },
  {
    accessorKey: "md",
    enableSorting: false,
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Metadata" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("md") as Record<string, any>;
      if (value === undefined) {
        return <Minus className="h-4 w-4 text-muted-foreground/50" />;
      }
      return (
        <div className="max-w-[240px] pointer-events-none">
          <MarkdownCodeBlock lang="json" text={JSON.stringify(value, null, 2)}/>
        </div>
      )
    }
  },
  {
    accessorKey: "document_md",
    enableSorting: false,
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Document Metadata" />
    ),
    cell: ({ row }) => {
      const value = row.getValue("document_md") as Record<string, any>;
      if (value === undefined) {
        return <Minus className="h-4 w-4 text-muted-foreground/50" />;
      }
      return (
        <div className="max-w-[240px] pointer-events-none">
          <MarkdownCodeBlock lang="json" text={JSON.stringify(value, null, 2)}/>
        </div>
      )
    }
  },
  // {
  //   accessorKey: "size_bytes",
  //   header: ({ column }) => (
  //     <DataTableColumnHeader column={column} title="Size" />
  //   ),
  //   cell: ({ row }) => {
  //     const value = row.getValue("size_bytes");
  //     if (typeof value === "undefined") {
  //       return <Minus className="h-4 w-4 text-muted-foreground/50" />;
  //     }
  //     if (typeof value === "number") {
  //       const file_size_string = file_size_as_string(value, false);
  //       const order_string = file_size_as_string(value, true);
  //       const colors = getStatusColor(parseInt(order_string));
  //       return <div className={`${colors.text} font-mono`}>{file_size_string}</div>;
  //     }
  //     return <div className="text-muted-foreground">{`${value}`}</div>;
  //   }
  // },
  // {
  //   accessorKey: "collection_type",
  //   header: ({ column }) => (
  //     <DataTableColumnHeader column={column} title="Type" />
  //   ),
  //   cell: ({ row }) => {
  //     const value = row.getValue("collection_type") as string;
  //     return <div className="font-mono">{value}</div>;
  //   },
  //   enableSorting: true,
  //   sortingFn: "alphanumeric"
  // },
  // {
  //   // TODO: make it a type of MethodSchema!
  //   accessorKey: "finished_processing",
  //   header: "Processing",
  //   // enableMultiSort: true,
  //   enableSorting: true,
  //   filterFn: "auto",
  //   cell: ({ row }) => {
  //     const value = row.getValue("finished_processing") as boolean;
  //     return (
  //       <>
  //         {value ? (
  //           <p className="text-green-500">Done</p>
  //         ):(
  //           <div>
  //             <div className="h-4 w-[90px] bg-accent flex items-center space-x-1 rounded-full px-2">
  //               <LucideLoader2 className="w-3 h-3 text-primary animate-spin" />
  //               <p className="text-xs">Processing</p>
  //             </div>
  //           </div>
  //         )}
  //       </>
  //     );
  //   }
  // }
];


