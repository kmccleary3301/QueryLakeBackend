"use client";
import { componentMetaDataType, displayMapping } from "@/types/toolchain-interface";
import { useToolchainContextAction } from "@/app/app/context-provider";
import BounceLoader from "react-spinners/BounceLoader";

export const METADATA : componentMetaDataType = {
  label: "Active Node Indicator",
  type: "Display",
  category: "Debugging",
  description: "Displays the current active node in the toolchain while running.",
};

type documentEmbeddingSpecialFields1 ={
  collection_type: string;
  document_id: string;
  document_chunk_number: number;
  document_integrity: string;
  parent_collection_hash_id: string;
} | {
  website_url: string;
}
type documentEmbeddingSpecialFields2 = { headline: string; cover_density_rank: number; } | {}
type documentEmbeddingSpecialFields3 = { rerank_score: number; } | {}

export type DocumentEmbeddingDictionary = {
	id: string;
	document_name: string;
	private: boolean;
	text: string;
  website_url?: string;
  rerank_score?: number;
} & documentEmbeddingSpecialFields1 & documentEmbeddingSpecialFields2 & documentEmbeddingSpecialFields3

export type chatEntry = {
	role?: "user" | "assistant",
	content: string,
	sources?: DocumentEmbeddingDictionary[]
}

export default function CurrentEventDisplay({
	configuration,
  demo = false
}:{
	configuration: displayMapping,
  demo?: boolean
}) {
	
	const { currentEvent } = useToolchainContextAction();

  return (
    <>
      {(currentEvent !== undefined || demo) && (
        <div className="w-auto h-11 flex flex-row justify-center gap-2">
          <div className="h-auto flex flex-col justify-center">
            <div className="">
              <BounceLoader size={20} color="rgb(20 184 166)" className="h-2 w-2 text-primary"/>
            </div>
          </div>
          <p className="h-auto flex flex-col justify-center">{
            demo ?
            "Running Event Node: sample_node" :
            `Running Event Node: ${currentEvent}`
          }</p>
        </div>
      )}
    </>
	);
}