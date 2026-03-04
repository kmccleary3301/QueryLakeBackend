"use client";
import { useState, useEffect, useRef, memo, useCallback, useMemo } from "react";
import { marked, TokensList, Token, Tokens } from 'marked';
import MarkdownTextSplitter from "./markdown-text-splitter";
import MarkdownCodeBlock from "./markdown-code-block";
import stringHash from "@/hooks/stringHash";
import MarkdownTable from "./markdown-table";
import sanitizeMarkdown from "@/components/markdown/sanitizeMarkdown";
import "./prose.css"
import { cn } from "@/lib/utils";
import { markdownRenderingConfig } from "./configs";
import MarkdownTypeRouter from "./markdown_type_router";

type MarkdownRendererProps = {
  input: string,
  transparentDisplay?: boolean,
  disableRender?: boolean,
  finished: boolean,
};

type MarkdownMapComponentProps = {
  token: Token,
  unProcessedText: string,
  key?: number,
  padLeft?: boolean,
  disableHeadingPaddingTop?: boolean,
  finished: boolean,
}

type MarkdownMapComponentErrorProps = {
  type: string
}

function MarkdownMapComponentError(props : MarkdownMapComponentErrorProps) {
  return (
    <div className="bg-red-600 rounded-lg items-center">
      <p className="text-lg text-red-600 p-2 text-center">
        {"Unfilled Markdown Component: "+props.type}
      </p>
    </div>
  );
}


const MarkdownRenderer = memo(function MarkdownRenderer({
  className = "",
  unpacked = false,
  input,
  transparentDisplay,
  disableRender = false,
  finished,
  config,
  list_in_block = false,
} : {
  className?: string,
  unpacked?: boolean,
  input: string,
  transparentDisplay?: boolean,
  disableRender?: boolean,
  finished: boolean,
  config: markdownRenderingConfig,
  list_in_block?: boolean
}) {
  const lexer = useMemo(() => new marked.Lexer(), []);
  // const lexed_input : Token[] = lexer.lex(sanitizeMarkdown(input));

  const [tokens, setTokens] = useState<Token[]>([]);

  const [unrenderedText, setUnrenderedText] = useState<string>("");
  const unusedRecentText = useRef<string>("");

  const last_render_time = useRef<number>(0);
  const rerender_timeout = useRef<NodeJS.Timeout>();
  const RENDER_INTERVAL_MS = 10;
  const USE_OLD_METHOD = false;


  const bufferedText = useRef<string>("");

  useEffect(() => {
    // ... (USE_OLD_METHOD check) ...

    // Throttling logic
    const time_until_next_render = 
        Math.max(0, RENDER_INTERVAL_MS - (Date.now() - last_render_time.current));
    
    if (rerender_timeout.current) {
      clearTimeout(rerender_timeout.current);
    }

    const getTimeout = setTimeout(() => {
      last_render_time.current = Date.now(); 
      
      // --- THIS IS THE BOTTLENECK ---
      // Lex our tokens.
      const new_tokens = lexer.lex(sanitizeMarkdown(input)); // Expensive operation!
      // ------------------------------
      
      // Update the tokens.
      setTokens(new_tokens); // Triggers re-render
      
    }, time_until_next_render);

    rerender_timeout.current = getTimeout;

  }, [input, lexer]); // Runs every time input changes

  return (
    <>
      {(disableRender)?(
        <>
          {input.split('\n').map((line, i) => (
            <p className="prose" key={i}>
              {line}
            </p>
          ))}
        </>
      ):(
        <>
        {unpacked?(
          <>
            {(
              // (unrenderedText.length > 0)?
              // [...tokens.slice(0, tokens.length-1), {
              //   ...tokens[tokens.length-1],
              //   raw: tokens[tokens.length-1].raw + unrenderedText,
              // }] :
              // tokens
              tokens
            ).map((v : Token, k : number) => (
              <MarkdownTypeRouter
                className={
                  (tokens[0].type === "list" && v.type === "list")?
                    (tokens[0].ordered) ?
                      "ml-[1.25rem]" : 
                      "ml-[1.25rem]" :
                    (list_in_block) ?
                      "ml-[1.25rem]" :
                      ""
                }
                key={k} 
                finished={finished}
                token={v} 
                unProcessedText={""}
                config={config}
              />
            ))}
          </>
        ):(
          <div className={cn("prose markdown text-sm text-theme-primary space-y-3 flex flex-col", className)}>
            <MarkdownRenderer 
              className={className}
              unpacked={true}
              input={input} 
              transparentDisplay={transparentDisplay}
              disableRender={disableRender}
              finished={finished}
              config={config}
            />
          </div>
        )}
        </>
      )}
    </>
  );
});

export default MarkdownRenderer;
