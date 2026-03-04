"use client";

import { DataTableInfinite } from "@/components/custom/data_table_infinite/data-table-infinite";
import { useContextAction } from "@/app/context-provider";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";
import { columnFilterSchema, columns, ColumnSchema, columnSchema, InfiniteQueryMeta, searchParamsParser, searchParamsSerializer } from "./columns";
// import { DataFetcher } from "./query-options";
import { fetchCollection } from "@/hooks/querylakeAPI";
import craftUrl from "@/hooks/craftUrl";

import axios from 'axios';
import { Button } from "@/components/ui/button"
import {
  fetch_collection_document_type,
  openDocument,
} from "@/hooks/querylakeAPI";
// import { useContextAction } from "@/app/context-provider";
// import craftUrl from "@/hooks/craftUrl";
import { useRouter } from 'next/navigation';
import "./spin.css";
import { ColumnDef, Table } from "@tanstack/react-table";
import { DataTableColumnHeader } from "@/components/data-table/data-table-column-header";
import { CollectionDataTableSheetDetails, CollectionSheetDetailsContent } from "./data-table-sheet-details";
import { useParams } from "next/navigation";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { filterFields as defaultFilterFields } from "./constants";
import { CollectionSidebar } from "./collection-sidebar";
import { userDataType } from "@/types/globalTypes";
import { infiniteQueryOptions, keepPreviousData } from "@tanstack/react-query";
import { SearchParamsType } from "./columns";
import LegacyNotice from "@/components/legacy/legacy-notice";

type SearchParams = {
  auth: string;
  collection_ids: string[];
  query: string;
  offset: number;
  limit: number;
  table: string;
  sort_by?: string;
  sort_dir?: "ASC" | "DESC";
}

const LAST_WORKSPACE_KEY = "ql_last_workspace";

// export type InfiniteQueryMeta = {
//   totalRowCount: number;
//   filterRowCount: number;
//   totalFilters: MakeArray<ColumnSchema>;
//   currentPercentiles: Record<Percentile, number>;
//   chartData: { timestamp: number; [key: string]: number }[];
// };

export type DataFetcher = (params: SearchParams) => Promise<{
  data: ColumnSchema[];
  meta: InfiniteQueryMeta;
}>;

// const COLLECTION_ID = "wAloo9uVIwU9IhidVvU2MR0JXKOWi5A6";


const dataOptions = (
  search: SearchParamsType,
  auth: string,
  collection_id: string,
  fetcher: DataFetcher,
) => {
  return infiniteQueryOptions({
    queryKey: ["data-table", searchParamsSerializer({ ...search })],
    queryFn: async ({ pageParam = 0 }) => {
      const start = (pageParam as number) * search.size;
      console.log("DataOptions Params", search);
      const searchParams: SearchParams = {
        auth,
        collection_ids: [collection_id],
        query: "",
        offset: start,
        limit: search.size,
        table: 'document',
        sort_by: search.sort ? `${search.sort.id}` : "file_name",
        sort_dir: search.sort ? (search.sort.desc ? "DESC" : "ASC") : undefined,
      };
      return fetcher(searchParams);
    },
    initialPageParam: 0,
    getNextPageParam: (_lastGroup, groups) => groups.length,
    refetchOnWindowFocus: false,
    placeholderData: keepPreviousData,
  });
};

const defaultFetcher: DataFetcher = async (params) => {
  console.log("defaultFetcher Params", params);
  const response = await fetch("/api/search_bm25", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });
  const result = await response.json();
  console.log("defaultFetcher Result", result);

  return Promise.resolve({
    data: result.result,
    meta: {
      totalRowCount: 1000,
      filterRowCount: result.total + 10,
    } as InfiniteQueryMeta
  });
};

type uploading_file_type = {
  title: string,
  progress: number,
}

type collection_mode_type = "create" | "edit" | "view" | undefined;

export default function Page() {
  const { userData, refreshCollectionGroups } = useContextAction();
  const [totalDBRowCount, setTotalDBRowCount] = useState(0);

  const resolvedParams = useParams() as {
    slug: string[],
  };

  const router = useRouter();
  const [redirecting, setRedirecting] = useState(false);

  const collection_mode_immediate = 
  (
    ["create", "edit", "view"].indexOf(
    resolvedParams["slug"][0]) 
    > -1
  ) ? 
    resolvedParams["slug"][0] as collection_mode_type :
    undefined;
  
  const [CollectionMode, setCollectionMode] = useState<collection_mode_type>(collection_mode_immediate);
  const [collectionTitle, setCollectionTitle] = useState<string>("");
  const [collectionDescription, setCollectionDescription] = useState<string>("");
  const [collectionDocuments, setCollectionDocuments] = useState<fetch_collection_document_type[]>([]);
  const [collectionIsPublic, setCollectionIsPublic] = useState<boolean>(false);
  const [collectionOwner, setCollectionOwner] = useState<string>("personal");
  const [uploadingFiles, setUploadingFiles] = useState<uploading_file_type[]>([]);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[] | null>(null);
  const [dataRowsProcessed, setDataRowsProcessed] = useState<ColumnSchema[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  const legacyCollectionId = resolvedParams["slug"]?.[1];
  const legacyWorkspacePath = legacyCollectionId
    ? `/collections/${legacyCollectionId}`
    : "/collections";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const lastWorkspace = window.localStorage.getItem(LAST_WORKSPACE_KEY);
    if (!lastWorkspace) return;
    const mode = resolvedParams["slug"]?.[0];
    const collectionId = resolvedParams["slug"]?.[1];
    if (mode === "create") {
      setRedirecting(true);
      router.replace(`/w/${lastWorkspace}/collections/new`);
      return;
    }
    if (mode === "view" && collectionId) {
      setRedirecting(true);
      router.replace(`/w/${lastWorkspace}/collections/${collectionId}`);
    }
  }, [router, resolvedParams]);
  
  const fetchCollectionCallback = useCallback(() => {
    if (!userData?.auth) return;
    fetchCollection({
      auth: userData.auth,
      collection_id: resolvedParams["slug"][1],
      onFinish: (data) => {
        console.log("Collection Data", data);
        if (data === undefined) { return; }
        setCollectionTitle(data.title);
        setCollectionDescription(data.description);
        setCollectionIsPublic(data.public);
        setTotalDBRowCount(data.document_count);
        setCollectionOwner(data.owner);
      }
    });
  }, [resolvedParams, userData?.auth]);


  // Keep refreshing collection documents every 5s if they are still processing
  useEffect(() => { 
    let documents_processing = false;
    collectionDocuments.forEach(doc => {
      if (!doc.finished_processing) {
        documents_processing = true;
      }
    });
    if (documents_processing) {
      setTimeout(() => {
        // TODO: update this to fetch documents where finished_processing is false
      }, 5000)
    }
  }, [collectionDocuments]);


  useEffect(() => {
    if ( userData?.auth !== undefined) {
      if (CollectionMode === "edit" || CollectionMode === "view") {
        // setCollectionMode(collection_mode_immediate)
        fetchCollectionCallback();
      }
    }
  }, [CollectionMode, fetchCollectionCallback, userData?.auth])

  const [search] = useQueryStates(searchParamsParser);
  const { data, isFetching, isLoading, fetchNextPage } = useInfiniteQuery(
    dataOptions(
      search,
      userData?.auth as string,
      resolvedParams["slug"][1],
      defaultFetcher
    )
  );

  const flatData = useMemo(
    () => data?.pages?.flatMap((page) => page.data ?? []) ?? [],
    [data?.pages]
  );

  useEffect(() => {
    console.log("Data", data);
  }, [data]);


  const lastPage = data?.pages?.[data?.pages.length - 1];
  const filterDBRowCount = lastPage?.meta?.filterRowCount;
  const totalFetched = flatData?.length;

  const { sort, start, size, id, ...filter } = search;
  
  useEffect(() => {
    setDataRowsProcessed(flatData);
  }, [flatData, uploadingFiles, pendingUploadFiles]);


  const filterFields = useMemo(
    () =>
      defaultFilterFields.map((field) => {
        // if (
        //   field.value === "latency" ||
        //   field.value === "timing.dns" ||
        //   field.value === "timing.connection" ||
        //   field.value === "timing.tls" ||
        //   field.value === "timing.ttfb" ||
        //   field.value === "timing.transfer" ||
        //   field.value === "status"
        // ) {
        //   field.options =
        //     totalFilters?.[field.value].map((value) => ({
        //       label: `${value}`,
        //       value,
        //     })) ?? field.options;
        // }
        // if (field.value === "host" || field.value === "pathname") {
        //   field.options =
        //     totalFilters?.[field.value].map((value) => ({
        //       label: `${value}`,
        //       value,
        //     })) ?? field.options;
        // }
        return field;
      }),
    []
  );

  if (redirecting) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
        Redirecting...
      </div>
    );
  }

  return (
    <div 
      className="w-full h-[calc(100vh)] absolute flex flex-row justify-center overflow-hidden"
      style={{
        "--container-width": "100%"
      } as React.CSSProperties}
    >
      <div className="absolute left-4 right-4 top-4 z-50">
        <LegacyNotice
          title="Legacy collections UI"
          description="This is the legacy collections experience. Use the new workspace UI for the recommended collections flow."
          workspacePath={legacyWorkspacePath as `/${string}`}
          ctaLabel="Open in workspace Collections"
        />
      </div>
      {/* <ScrollArea className="w-full"> */}
        <div className="flex flex-row w-[var(--container-width)] justify-center overflow-hidden">
        <SidebarProvider open={sidebarOpen} className="w-full">
          <SidebarInset>
          
            <div className="flex flex-col overflow-hidden w-full sticky">
              <DataTableInfinite
                className="overflow-hidden w-full"
                columns={columns}
                data={dataRowsProcessed}
                // data={flatData}
                totalRows={totalDBRowCount}
                // filterRows={filterDBRowCount}
                filterRows={totalDBRowCount}
                totalRowsFetched={totalFetched}
                defaultColumnFilters={Object.entries(filter)
                  .map(([key, value]) => ({
                    id: key,
                    value,
                  }))
                  .filter(({ value }) => value ?? undefined)}
                filterFields={filterFields}
                defaultColumnSorting={sort ? [sort] : undefined}
                searchColumnFiltersSchema={columnFilterSchema}
                searchParamsParser={searchParamsParser}
                defaultRowSelection={search.id ? { [search.id]: true } : undefined}
                isFetching={isFetching}
                isLoading={isLoading}
                fetchNextPage={fetchNextPage}
                rowEntrySidebarComponent={(props: {selectedRow: ColumnSchema | undefined, table: Table<ColumnSchema>}) => {
                  return (
                    <CollectionDataTableSheetDetails
                      // TODO: make it dynamic via renderSheetDetailsContent
                      title={(props.selectedRow as ColumnSchema | undefined)?.file_name}
                      titleClassName="font-mono"
                      table={props.table}
                      onDownload={() => {
                        if (!props.selectedRow) return;
                        openDocument({
                          auth: userData?.auth as string,
                          document_id: props.selectedRow?.id
                        });
                      }}
                      viewChunks={() => {
                        window.open(`/file/view/${props.selectedRow?.id}?sort=document_chunk_number.asc`);
                      }}
                    >
                      <CollectionSheetDetailsContent
                        data={props.selectedRow as ColumnSchema}
                        filterRows={totalDBRowCount}
                      />
                    </CollectionDataTableSheetDetails>
                  )
                }}
                setControlsOpen={setSidebarOpen}
              />
            </div>
          </SidebarInset>
          <CollectionSidebar
            collapsible="motion" 
            side="right"
            user_auth={userData as userDataType}
            collection_id={resolvedParams["slug"][1]}
            collection_is_public={collectionIsPublic}
            set_collection_is_public={setCollectionIsPublic}
            collection_owner={collectionOwner}
            set_collection_owner={setCollectionOwner}
            collection_name={collectionTitle}
            set_collection_name={setCollectionTitle}
            collection_description={collectionDescription}
            set_collection_description={setCollectionDescription}
            add_files={(files) => {
              
            }}
          />
        </SidebarProvider>
          
        </div>
      {/* </ScrollArea> */}
    </div>
  );
}
