"use client";

import type {
  ColumnDef,
  ColumnFiltersState,
  RowSelectionState,
  SortingState,
  Table as TTable,
  VisibilityState,
} from "@tanstack/react-table";
import {
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import * as React from "react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/custom/table";
import { DataTableFilterCommand } from "@/components/data-table/data-table-filter-command";
// import { ColumnSchema, columnFilterSchema } from "./schema";
import type { DataTableFilterField } from "@/components/data-table/types";
import { DataTableToolbar } from "@/components/data-table/data-table-toolbar"; // TODO: check where to put this
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";
import { useQueryStates, UseQueryStatesKeysMap } from "nuqs";
// import { searchParamsParser } from "./search-params";
import { type FetchNextPageOptions } from "@tanstack/react-query";
import { LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatCompactNumber } from "@/lib/format";
import { inDateRange, arrSome } from "@/lib/table/filterfns";
import { z } from "zod";
import { ScrollArea } from "@/components/ui/scroll-area";
import useResizeObserver from "@react-hook/resize-observer";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"
import { set } from "date-fns";

const useSize = (target : React.RefObject<HTMLDivElement>) => {
  const [size, setSize] = React.useState<DOMRect>()

  React.useLayoutEffect(() => {
		if (target.current !== null) {
			setSize(target.current.getBoundingClientRect())
		}
  }, [target])

  // Where the magic happens
  useResizeObserver(target, (entry) => setSize(entry.contentRect))
  return size
}

// TODO: add a possible chartGroupBy
export interface DataTableInfiniteProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  searchColumnFiltersSchema: z.AnyZodObject;
  searchParamsParser: UseQueryStatesKeysMap<any>;
  defaultColumnFilters?: ColumnFiltersState;
  searchColumnFilters?: ColumnFiltersState;
  defaultColumnSorting?: SortingState;
  defaultRowSelection?: RowSelectionState;
  defaultColumnVisibility?: VisibilityState;
  arrayColumns?: string[]; // New prop for columns that contain arrays
  filterFields?: DataTableFilterField<TData>[];
  totalRows?: number;
  filterRows?: number;
  totalRowsFetched?: number;
  isFetching?: boolean;
  isLoading?: boolean;
  fetchNextPage: (options?: FetchNextPageOptions | undefined) => void;
  className?: string;
  onSelectRow?: (row: TData) => void;
  onToggleControls?: (opened: boolean) => void;
  rowEntrySidebarComponent?: (props: {selectedRow: TData | undefined, table: TTable<TData>}) => React.ReactNode;
  collectionSidebarTrigger?: React.ReactNode;
  setControlsOpen?: React.Dispatch<React.SetStateAction<boolean>>;
}

export function DataTableInfinite<TData, TValue>({
  columns,
  data,
  searchColumnFiltersSchema,
  searchParamsParser,
  defaultColumnFilters = [],
  defaultColumnSorting = [],
  defaultRowSelection = {},
  defaultColumnVisibility = {},
  arrayColumns = [],
  filterFields = [],
  isFetching,
  isLoading,
  fetchNextPage,
  totalRows = 0,
  filterRows = 0,
  totalRowsFetched = 0,
  className = "",
  onSelectRow = () => {},
  onToggleControls = () => {},
  rowEntrySidebarComponent: sidebarComponent,
  setControlsOpen,
}: DataTableInfiniteProps<TData, TValue>) {
  const [columnFilters, setColumnFilters] =
    React.useState<ColumnFiltersState>(defaultColumnFilters);
  const [sorting, setSorting] =
    React.useState<SortingState>(defaultColumnSorting);
  const [rowSelection, setRowSelection] =
    React.useState<RowSelectionState>(defaultRowSelection);
  const [columnOrder, setColumnOrder] = useLocalStorage<string[]>(
    "data-table-column-order",
    []
  );
  const [columnVisibility, setColumnVisibility] =
    useLocalStorage<VisibilityState>("data-table-visibility", defaultColumnVisibility);
  const [controlsOpen, setControlsOpenInner] = useLocalStorage(
    "data-table-controls",
    true
  );
  const topBarRef = React.useRef<HTMLDivElement>(null);
  const [topBarHeight, setTopBarHeight] = React.useState(0);
  const [_, setSearch] = useQueryStates(searchParamsParser);

  React.useEffect(() => {
    const observer = new ResizeObserver(() => {
      const rect = topBarRef.current?.getBoundingClientRect();
      if (rect) {
        setTopBarHeight(rect.height);
      }
    });

    const topBar = topBarRef.current;
    if (!topBar) return;

    observer.observe(topBar);
    return () => observer.unobserve(topBar);
  }, [topBarRef]);

  const [bottomScrollRefresher, setBottomScrollRefresher] = React.useState(true);


  React.useEffect(() => {
    // if (typeof window === "undefined") return;
    if (!bottomScrollRefresher) return;

    // TODO: add a threshold for the "Load More" button
    const onPageBottom =
      window.innerHeight + Math.round(window.scrollY) >=
      document.body.offsetHeight;
    if (onPageBottom && !isFetching && totalRowsFetched < filterRows) {
      console.log("Fetching next page...");
      fetchNextPage();
      setBottomScrollRefresher(false);
    }
  }, [fetchNextPage, isFetching, filterRows, totalRowsFetched, bottomScrollRefresher]);


  const table = useReactTable({
    data,
    columns,
    state: {
      columnFilters,
      sorting,
      columnVisibility,
      rowSelection,
      columnOrder,
    },
    enableMultiRowSelection: false,
    getRowId: (row: any) => {
      // Try different unique identifiers in order of preference
      return row?.id || row?.uuid || String(Math.random());
    },
    onColumnVisibilityChange: setColumnVisibility,
    onColumnFiltersChange: setColumnFilters,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnOrderChange: setColumnOrder,
    getSortedRowModel: getSortedRowModel(),
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: (table: TTable<TData>, columnId: string) => () => {
      const map = getFacetedUniqueValues<TData>()(table, columnId)();
      if (arrayColumns.includes(columnId)) {
        const rowValues = table
          .getGlobalFacetedRowModel()
          .flatRows.map((row) => row.getValue(columnId) as string[]);
        for (const values of rowValues) {
          for (const value of values) {
            const prevValue = map.get(value) || 0;
            map.set(value, prevValue + 1);
          }
        }
      }
      return map;
    },
    filterFns: { inDateRange, arrSome },
  });

  React.useEffect(() => {
    const columnFiltersWithNullable = filterFields.map((field) => {
      const filterValue = columnFilters.find(
        (filter) => filter.id === field.value
      );
      if (!filterValue) return { id: field.value, value: null };
      return { id: field.value, value: filterValue.value };
    });

    const search = columnFiltersWithNullable.reduce((prev, curr) => {
      prev[curr.id as string] = curr.value;
      return prev;
    }, {} as Record<string, unknown>);

    setSearch(search);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columnFilters]);

  React.useEffect(() => {
    setSearch({ sort: sorting?.[0] || null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sorting]);

  const selectedRow = React.useMemo(() => {
    const selectedRowKey = Object.keys(rowSelection)?.[0];
    return table
      .getCoreRowModel()
      .flatRows.find((row) => row.id === selectedRowKey);
  }, [rowSelection, table]);

  // FIXME: cannot share a uuid with the sheet details
  React.useEffect(() => {
    if (Object.keys(rowSelection)?.length && !selectedRow) {
      setSearch({ uuid: null });
      setRowSelection({});
    } else {
      setSearch({ uuid: Object.keys(rowSelection)?.[0] || null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    console.log("SELECTED ROW CALLED:", selectedRow);

  }, [rowSelection, selectedRow, setSearch]);

  React.useEffect(() => {
    console.log("totalRowsFetched", totalRowsFetched);
    console.log("filterRows", filterRows);
    console.log("totalRows", totalRows);
  }, [totalRowsFetched, filterRows, totalRows]);

  React.useEffect(() => {
    if (setControlsOpen === undefined) return;
    setControlsOpen(controlsOpen);
  }, [controlsOpen, setControlsOpen]);

  return (
    <>
      <ScrollArea 
        className={cn("flex w-full h-screen flex-col sm:flex-row pr-2 overflow-hidden sticky scrollbar-hide", className)}
        onScroll={(e) => {
          const scrollTop = e.currentTarget.scrollTop;
          const scrollHeight = e.currentTarget.scrollHeight;
          const clientHeight = e.currentTarget.clientHeight;
          const scrollPercentage = (scrollTop / (scrollHeight - clientHeight)) * 100;
          const scrollDelta = (scrollHeight-clientHeight)-scrollTop;
          if (scrollDelta < 10 && !isFetching && totalRowsFetched < filterRows) {
            console.log("Fetching next page...");
            fetchNextPage();
          }
        }}
      >
        <div
          className={cn(
            "flex max-w-full flex-1 flex-col sm:border-l border-border overflow-clip",
            // Chrome issue
            controlsOpen &&
              "max-w-[calc(100vw_-_58px)] sm:max-w-[calc(100vw_-_208px)] md:max-w-[calc(100vw_-_288px)]"
          )}
        >
          <div
            ref={topBarRef}
            className={cn(
              "flex flex-col gap-4 bg-background p-2",
              "z-10 pb-4 sticky top-0"
            )}
          >
            <DataTableFilterCommand
              table={table}
              schema={searchColumnFiltersSchema}
              filterFields={filterFields}
              isLoading={isFetching || isLoading}
            />
            <DataTableToolbar
              table={table}
              controlsOpen={controlsOpen}
              setControlsOpen={setControlsOpenInner}
              isLoading={isFetching || isLoading}
              enableColumnOrdering={true}
            />
          </div>
          <div className="z-0">
            <Table containerClassName="overflow-clip">
              <TableHeader
                className="sticky bg-muted z-20"
                style={{ top: `${topBarHeight}px` }}
              >
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow
                    key={headerGroup.id}
                    className="hover:bg-transparent"
                  >
                    {headerGroup.headers.map((header) => {
                      return (
                        <TableHead
                          key={header.id}
                          className={
                            header.column.columnDef.meta?.headerClassName
                          }
                        >
                          {header.isPlaceholder
                            ? null
                            : flexRender(
                                header.column.columnDef.header,
                                header.getContext()
                              )}
                        </TableHead>
                      );
                    })}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {/* FIXME: should be getRowModel() as filtering */}
                {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row, row_index) => (
                    <TableRow
                      // key={row.id}
                      key={row_index}
                      data-state={row.getIsSelected() && "selected"}
                      onClick={() => row.toggleSelected()}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          className={
                            cell.column.columnDef.meta?.headerClassName
                          }
                        >
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={columns.length}
                      className="h-24 text-center"
                    >
                      No results.
                    </TableCell>
                  </TableRow>
                )}
                <TableRow className="hover:bg-transparent data-[state=selected]:bg-transparent">
                  <TableCell colSpan={columns.length} className="text-center">
                    {totalRowsFetched < filterRows ||
                    !table.getCoreRowModel().rows?.length ? (
                      <Button
                        disabled={isFetching || isLoading}
                        onClick={() => fetchNextPage()}
                        size="sm"
                        variant="outline"
                      >
                        {isFetching ? (
                          <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                        ) : null}
                        Load More
                      </Button>
                    ) : (
                      <p className="text-muted-foreground text-sm">
                        No more data to load (total:{" "}
                        <span className="font-medium font-mono">
                          {formatCompactNumber(totalRows)}
                        </span>{" "}
                        rows)
                      </p>
                    )}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </div>
      </ScrollArea>
      {(sidebarComponent !== undefined) && (
        sidebarComponent({ selectedRow: selectedRow?.original, table: table })
      )}
    </>
  );
}
