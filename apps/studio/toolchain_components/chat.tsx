"use client";
import { retrieveValueFromObj } from "@/hooks/toolchain-session";
import { Skeleton } from "@/components/ui/skeleton";
import { componentMetaDataType, configEntriesMap, displayMapping } from "@/types/toolchain-interface";
import { ElementType, Fragment, useCallback, useEffect, useRef, useState } from "react";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";
import { useToolchainContextAction } from "@/app/app/context-provider";
import { useContextAction } from "@/app/context-provider";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowUpRightFromCircle, Copy, FileText } from "lucide-react";
import { toast } from "sonner";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import Link from "next/link";
import { openDocument } from "@/hooks/querylakeAPI";
import { QueryLakeLogoSvg } from "@/components/logo";
import { MARKDOWN_CHAT_SAMPLE_TEXT } from "@/components/markdown/demo-text";
import { cn } from "@/lib/utils";
import { ScrollArea, ScrollAreaHorizontal } from "@/components/ui/scroll-area";
import { CHAT_RENDERING_STYLE, markdownRenderingConfig } from "@/components/markdown/configs";
import { textSegment } from "@/components/markdown/markdown-text-splitter";
import { parse } from "path";
import { AnimatePresence, motion } from "framer-motion";
import SmoothHeightDiv from "@/components/manual_components/smooth-height-div";
import TextWithTooltip from "@/components/custom/text-with-tooltip";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import "./shimmer_1.css"
import { handleCopy } from "@/components/markdown/markdown-code-block";


export const METADATA : componentMetaDataType = {
  label: "Chat",
  type: "Display",
  category: "Text Display",
  description: "A user and assistant chat display component that renders markdown, similar to OpenAI's ChatGPT."
};

export const DEMO_DATA = [
  {role: "user", "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."},
  {role: "assistant", "content": MARKDOWN_CHAT_SAMPLE_TEXT},
] as chatEntry[];

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

const AnimatePresenceFixedType = AnimatePresence as ElementType;

export type DocumentEmbeddingDictionary = {
	id: string;
	document_name: string;
	private: boolean;
	text: string;
  website_url?: string;
  rerank_score?: number;
  document_id: string;
} & documentEmbeddingSpecialFields1 & documentEmbeddingSpecialFields2 & documentEmbeddingSpecialFields3

export type chatEntry = {
	role?: "user" | "assistant",
  headline?: {search: string}[],
	content: string,
	sources?: DocumentEmbeddingDictionary[]
}

function InlineSource({
  sources,
  textSeg,
  user_auth
}:{
  sources: DocumentEmbeddingDictionary[],
  textSeg: textSegment,
  user_auth: string
}) {
  const parseSourceIndex = parseInt(textSeg.text) - 1;
  
  return (
    <>
      {((!isNaN(parseSourceIndex)) && (parseSourceIndex < sources.length) && (sources[parseSourceIndex] !== undefined)) ? (
        <HoverCard>
          <HoverCardTrigger asChild>
            <span style={{ transform: 'translateY(-2px)', display: 'inline-block', marginLeft: '2px' }}>
              <button className="text-primary/50 hover:text-primary inline-block bg-input rounded-full flex-row justify-center -translate-y-[2px]" style={{
                // paddingLeft: "0.1rem",
                fontSize: "0.65rem",
                height: "18px",
                minWidth: "18px",
              }}>
                <div className="w-auto flex flex-row justify-center">
                  <strong className="p-0 m-0 self-center leading-none">
                    {textSeg.text}
                  </strong>
                </div>
              </button>
            </span>
          </HoverCardTrigger>
          <HoverCardContent className="p-0 max-w-[320px] m-0" side="top">
            <h4 className="px-5 py-4 text-base break-words w-[320px]">
              <span>{sources[parseSourceIndex].document_name}</span>
              <Button variant={"transparent"} className="ml-2 p-0 w-8 h-8" onClick={() => {
                openDocument({
                  auth: user_auth,
                  document_id: sources[parseSourceIndex].document_id as string,
                })
              }}>
                <TooltipProvider disableHoverableContent delayDuration={50}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <ArrowUpRightFromCircle className="w-4 h-4 text-primary"/>
                    </TooltipTrigger>
                    <TooltipContent className="bg-popover opacity-100">Go to Document</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                  
              </Button>
            </h4>
            {sources[parseSourceIndex].rerank_score && (
              <p className="text-sm py-3">Relevance Score: {sources[parseSourceIndex].rerank_score?.toFixed(2)}</p>
            )}
            <div className="border-primary" style={{borderWidth: "0 0 0 0.2rem"}}>
            <ScrollArea className="h-[200px] pr-3 px-5">
            {(sources[parseSourceIndex].website_url !== undefined) ? (
              <Link href={sources[parseSourceIndex]?.website_url || ""} rel="noopener noreferrer" target="_blank">
                <Button variant={"ghost"} className="p-2 m-0 h-auto">
                  <div className="max-w-[260px]">
                    <p className="max-w-[260px] text-xs text-primary/50 whitespace-pre-wrap text-left overflow-wrap break-words">{sources[parseSourceIndex].text}</p>
                  </div>
                </Button>
              </Link>
            ):(
              <div className="justify-start pb-[20px]">
                <div className="" style={{marginLeft: "0.1rem" }}> {/* The left pad doesn't render for some reason */}
                  <MarkdownRenderer 
                    className="w-[280px]"
                    input={sources[parseSourceIndex].text} 
                    finished={false}
                    config={CHAT_RENDERING_STYLE}
                  />
                </div>
              </div>
            )}
            </ScrollArea>
            </div>
          </HoverCardContent>
        </HoverCard>
        
      ):(
        <HoverCard>
          <HoverCardTrigger asChild>
            <span style={{ transform: 'translateY(-2px)', display: 'inline-block' }}>
              <button className="text-primary/50 hover:text-primary inline-block bg-accent rounded-full flex-row justify-center -translate-y-[2px]" style={{
                fontSize: "0.65rem",
                height: "18px",
                minWidth: "18px",
              }}>
                <div className="w-auto flex flex-row justify-center">
                  <strong className="p-0 m-0 self-center leading-none">?</strong>
                </div>
              </button>
            </span>
          </HoverCardTrigger>
          <HoverCardContent className="px-5" side="top">
            <p>Unknown source cited: {textSeg.raw_text}</p>
          </HoverCardContent>
        </HoverCard>
      )}
    </>
  )
}


type source_compiled = {
  document_id: string;
  document_name: string;
  sources: DocumentEmbeddingDictionary[];
}

function SourcesBarSource({
  source,
  user_auth,
}:{
  source: source_compiled,
  user_auth: string,
}) {

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <div
          className={`rounded-lg px-2 py-1 h-[45px] w-[200px] bg-accent flex flex-row justify-start`}
          // style={{
          //   maxWidth: "120px",
          // }}
        >
          <div className="w-[35px] h-auto flex flex-col justify-start pt-[5px] pr-[5px]" style={{paddingTop: "5px"}}>
            <FileText className="w-6 h-6 text-primary"/>
          </div>
          <div className="w-[140px] flex flex-col justify-around overflow-hidden">
            <span className="text-primary text-xs text-wrap line-clamp-2 h-[5rem] text-ellipsis">
              {source.document_name}
            </span>
            <p className="text-xs opacity-30 text-primary/50">{source.sources.length} segments</p>
          </div>
        </div>
      </HoverCardTrigger>
      <HoverCardContent className="px-5 max-w-[320px]" side="top">
        {/* <h1 className="text-base">{source.document_name}</h1>
        {source.rerank_score && (
          <p className="text-sm py-3">Relevance Score: {source.rerank_score.toFixed(2)}</p>
        )} */}
        <ScrollArea className="h-[200px] pr-3">
        {/* {(source.website_url) ? (
          <Link href={source.website_url} rel="noopener noreferrer" target="_blank">
            <Button variant={"ghost"} className="p-2 m-0 h-auto">
              <div className="max-w-[260px]">
                <p className="max-w-[260px] text-xs text-primary/50 whitespace-pre-wrap text-left overflow-wrap break-words">{source.text}</p>
              </div>
            </Button>
          </Link>
        ):(
          <Button variant={"ghost"} className="p-2 m-0 h-auto" onClick={()=>{
            openDocument({
              auth: user_auth,
              document_id: source?.document_id as string,
            })
          }}>
            <div className="max-w-[260px]">
              <p className="max-w-[260px] text-xs text-primary/50 whitespace-normal text-left overflow-wrap break-word">{source.text}</p>
            </div>
          </Button>
        )} */}
        </ScrollArea>
      </HoverCardContent>
    </HoverCard>
  )
}


function SourcesBar({
  sources,
  user_auth,
  className = "",
}:{
  sources: DocumentEmbeddingDictionary[],
  user_auth: string,
  className?: string
}) {

  const [compiledSources, setCompiledSources] = useState<source_compiled[]>([]);

  useEffect(() => {
    // map by document_id
    let sources_map = new Map<string, source_compiled>();
    for (let source of sources) {
      if (sources_map.has(source.document_id)) {
        sources_map.get(source.document_id)?.sources.push(source);
      } else {
        sources_map.set(source.document_id, {
          document_id: source.document_id,
          document_name: source.document_name,
          sources: [source]
        });
      }
    }
    const new_sources_compiled = Array.from(sources_map.values());

    setCompiledSources(new_sources_compiled);
  }, [sources]);


  return (
    <div className="flex flex-row gap-2 overflow-x-scroll scrollbar-hide pt-2" style={{
      marginLeft: "2.75rem",
    }}>
      <AnimatePresenceFixedType>
        {compiledSources.map((source, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.3,
              delay: index * 0.1, // stagger effect
              ease: "easeOut"
            }}
          >
            <SourcesBarSource
              source={source}
              user_auth={user_auth}
            />
          </motion.div>
        ))}
      </AnimatePresenceFixedType>
    </div>
  )
}

function MarkdownSubComponent({
  text,
  config,
  disabled = false
}:{
  text: string,
  config: markdownRenderingConfig,
  disabled: boolean
}) {
  
  const [workingText, setWorkingText] = useState(text);
  const cachedText = useRef<string>("");
  const workingTextBuffer = useRef<string>("");
  const timeouts = useRef<NodeJS.Timeout[]>([]);

  const addToken = useCallback((delay: number) => {
    if (workingTextBuffer.current.length === 0) return;
    return setTimeout(() => {
      setWorkingText(workingText + workingTextBuffer.current.slice(0, 1));
      workingTextBuffer.current = workingTextBuffer.current.slice(1);
    }, delay);
  }, [workingText, workingTextBuffer]);

  useEffect(() => {
    for (let t of timeouts.current) {
      clearTimeout(t);
    }
    timeouts.current = [];
    
    if (cachedText.current != text.slice(0, cachedText.current.length)) {
      setWorkingText(text);
    } else {
      workingTextBuffer.current = text;
      const stagger_delay = 1 / workingTextBuffer.current.length;
      for (let i = 0; i < workingTextBuffer.current.length; i++) {
        const timeout = addToken(i * stagger_delay);
        if (timeout) timeouts.current.push(timeout);
      }
    }
    cachedText.current = text;
  }, [text, workingTextBuffer]);


  return (
    <MarkdownRenderer 
      className="ml-11 text-primary"
      disableRender={disabled}
      input={text} 
      finished={false}
      config={config}
      unpacked
    />
  )
}

export default function Chat({
	configuration,
  demo = false,
}:{
	configuration: displayMapping,
  demo?: boolean
}) {
	
	const { toolchainState, toolchainWebsocket } = useToolchainContextAction();
  const { userData } = useContextAction();

	const [currentValue, setCurrentValue] = useState<chatEntry | chatEntry[]>(
    demo ?
    DEMO_DATA :
    retrieveValueFromObj(toolchainState, configuration.display_route) as chatEntry | chatEntry[] || []
	);

	useEffect(() => {
		if (toolchainWebsocket?.current === undefined || demo) return;
    const newValue = retrieveValueFromObj(toolchainState, configuration.display_route) as chatEntry | chatEntry[] || [];
    // console.log("Chat newValue", JSON.parse(JSON.stringify(newValue)));
		setCurrentValue(newValue);
	}, [toolchainState, setCurrentValue, toolchainWebsocket, demo]);

  return (
    <>
      {currentValue && (
        <div className="flex flex-col gap-8 pb-2">
          <AnimatePresenceFixedType>
          {(Array.isArray(currentValue)?currentValue:[currentValue]).map((value, index) => (
            
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                // delay: index * 0.1, // stagger effect
                ease: "easeOut"
              }}
            >
            <div className="flex flex-col gap-0" key={index}>
              <div key={index} className="flex flex-row">
                <div className="w-11 h-8 pt-[5px]">
                  {(value.role === "user") && (
                    <div className={`rounded-full h-7 w-7 bg-primary-foreground text-primary`}>
                    <p className="w-full h-full text-center text-xs flex flex-col justify-center pt-[2px] select-none">
                      {userData?.username.slice(0, Math.min(2, userData?.username.length)).toUpperCase()}
                    </p>
                    </div>
                  )}

                  {(value.role === "assistant") && (
                    <div className="mt-1">
                      <QueryLakeLogoSvg className="w-7 h-7 text-primary"/>
                    </div>
                  )}
                  
                    
                </div>
                <p className="select-none h-7 text-primary/70">{(value.role === "user")?"You":"QueryLake"}</p>
              </div>
              <div> {/* The left pad doesn't render for some reason */}
                
              <div className={cn("max-w-full -mt-1.5")} style={{marginLeft: "2.75rem", marginTop: -5}}>
                {(value.headline && 
                  value.headline.length > 0 && 
                  value.headline[value.headline.length-1].search !== undefined &&
                  !(value.content && value.content.length > 0)
                ) && (
                  <p className="shimmer">
                    {value.headline[value.headline.length-1].search}
                  </p>
                )}
                <div className="flex flex-col gap-y-1 prose markdown">
                  <MarkdownSubComponent
                    disabled={(value.role === "user")}
                    text={(value || {}).content || ""}
                    config={{
                      ...CHAT_RENDERING_STYLE,
                      citation: (textSeg: textSegment) => (
                        <InlineSource
                          sources={(value || {}).sources || []}
                          textSeg={textSeg}
                          user_auth={userData?.auth as string}
                        />
                      )
                    }}
                  />
                </div>
              </div>
              </div>
              {(value.role === "assistant" && value.sources) && (
                <SourcesBar sources={value.sources} user_auth={userData?.auth as string} className="ml-[10px]"/>
              )}
              {(value.role === "assistant") && (
                <div className="w-full flex flex-row pl-11 pt-2" style={{marginLeft: "2.75rem"}}>
                  <Button variant={"ghost"} className="rounded-full p-0 h-8 w-8" onClick={() => handleCopy((value || {}).content || "")}>
                    <Copy className="w-4 h-4 text-primary"/>
                  </Button>
                </div>
              )}
            </div>
            </motion.div>
          ))}
          </AnimatePresenceFixedType>
        </div>
      )}
    </>
	);
}