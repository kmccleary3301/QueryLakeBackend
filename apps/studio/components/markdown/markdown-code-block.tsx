"use client";
import { ScrollArea, ScrollAreaHorizontal, ScrollBar } from '@/components/ui/scroll-area';
// import hljs from 'highlight.js';
// import { getHighlighter } from 'shiki';
import { useEffect, useState, useRef, useCallback } from "react";
import { fontConsolas } from '@/lib/fonts';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import * as Icon from 'react-feather';
import { toast } from 'sonner';
import { getLanguage, highlight } from '@/lib/shiki';
// import ScrollSection from '../manual_components/scrollable-bottom-stick/custom-scroll-section';
import { Copy } from 'lucide-react';
import { BundledLanguage } from 'shiki/langs';
import { useContextAction } from "@/app/context-provider";
// import { renderToHtml } from "shiki";
// import codeToHTML
// import { codeToHtml } from 'shiki/index.mjs';
// import { Lang } from 'shiki';

const code_styling = new Map<string, string>([
  ["hljs-keyword", "#2E80FF"],
  ["hljs-function", "#DCDCAA"],
  ["hljs-meta", "#CE9178"],
  ["hljs-comment", "#6A9955"],
  ["hljs-params", "#8CDCF0"],
  ["hljs-literal", "#DF3079"],
  ["hljs-string", "#CE9178"],
  ["hljs-title class_", "#32BBB0"],
  ["hljs-title function_", "#DCDCAA"],
  ["hljs-number", "#DF3079"],
  ["hljs-built_in", "#DCDCAA"],
  ["hljs-type", "#4EC9B0"],
  ["default", "#DCDCAA"],
]);

type MarkdownCodeBlockProps = {
  text : string,
  unProcessedText: string,
  lang: string,
  finished: boolean
}

type scoped_text = {
  scope: string[],
  content: string
};

export const handleCopy = (text: string) => {
  if (typeof window === 'undefined') {
    return Promise.reject('Window is not defined');
  }

  // Try the modern Clipboard API first
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text)
      .then(() => {
        toast("Copied to clipboard");
        return true;
      })
      .catch((err) => {
        console.error("Clipboard API failed:", err);
        // Fall back to document.execCommand method
        return fallbackCopy(text);
      });
  } else {
    // Use fallback for browsers without Clipboard API support
    return fallbackCopy(text);
  }
};

// Fallback copy method using execCommand
const fallbackCopy = (text: string): Promise<boolean> => {
  return new Promise((resolve) => {
    try {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      
      // Make the textarea out of viewport
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      
      const successful = document.execCommand('copy');
      document.body.removeChild(textArea);
      
      if (successful) {
        toast("Copied to clipboard");
        resolve(true);
      } else {
        toast("Failed to copy to clipboard");
        resolve(false);
      }
    } catch (err) {
      console.error("Fallback copy failed:", err);
      toast("Failed to copy to clipboard");
      resolve(false);
    }
  });
};

const DEBOUNCE_DELAY = 15; // ms - Adjust as needed

export default function MarkdownCodeBlock({
  className = "",
  text,
  unProcessedText = "",
  lang,
  finished = true,
}:{
  className?: string,
  text : string,
  unProcessedText?: string,
  lang: string,
  finished?: boolean
}){

  const { shikiTheme } = useContextAction();
  const [language, setLanguage] = useState<{value: BundledLanguage | "text", preview: string}>({value: "text", preview: "Text"});
  const [codeHTML, setCodeHTML] = useState<string>("");
  const [lineCount, setLineCount] = useState<number>(0);

  // --- Refs for Throttling ---
  const lastHighlightTimeRef = useRef<number>(0);
  const highlightTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Combine text based on finished state
  const currentFullCode = finished ? text : text + (unProcessedText || "");

  useEffect(() => {
    // --- Basic Setup ---
    setLineCount(currentFullCode.split("\n").length);
    const languageInfo = getLanguage(lang);
    setLanguage(languageInfo);

    // --- Handle Plain Text Case ---
    if (languageInfo.value === "text") {
      if (highlightTimeoutRef.current) clearTimeout(highlightTimeoutRef.current); // Clear pending highlight
      setCodeHTML(currentFullCode); // Display raw text directly
      lastHighlightTimeRef.current = 0; // Reset timer state
      return; // Exit early
    }

    // --- Throttling Logic ---
    // Calculate time needed until the next allowed highlight execution
    const timeSinceLast = Date.now() - lastHighlightTimeRef.current;
    const timeUntilNextHighlight = Math.max(0, DEBOUNCE_DELAY - timeSinceLast);

    // Always clear the *previous* timeout if one exists. This ensures only the
    // latest update request within the interval window will eventually fire.
    if (highlightTimeoutRef.current) {
      clearTimeout(highlightTimeoutRef.current);
    }

    // Set a new timeout to perform the highlight
    highlightTimeoutRef.current = setTimeout(() => {
      lastHighlightTimeRef.current = Date.now(); // Record execution time *when it starts*

      // Use the code captured when the timeout was scheduled
      const codeToHighlight = finished ? text : text + (unProcessedText || "");

      // Perform the async highlight
      highlight(codeToHighlight, shikiTheme.theme, languageInfo.value)
        .then((html) => {
          setCodeHTML(html);
        })
        .catch(error => {
          console.error("Shiki highlighting failed:", error);
          setCodeHTML(codeToHighlight); // Fallback to raw code on error
        });

    }, timeUntilNextHighlight); // Wait for the calculated delay

    // --- Cleanup Function ---
    return () => {
      if (highlightTimeoutRef.current) {
        clearTimeout(highlightTimeoutRef.current);
      }
    };

  // Dependencies: Re-run when code content, language, finished status, or theme changes.
  }, [currentFullCode, lang, finished, shikiTheme, text, unProcessedText]); // Include text/unProcessedText if needed for timeout closure

  return (
    <div className={cn(
      'not-prose rounded-lg flex flex-col font-consolas my-3 border border-input', 
      fontConsolas.className,
      className
    )} style={{
      backgroundColor: (shikiTheme.backgroundColor || "#000000"), 
      color: (shikiTheme.textColor || "#FFFFFF"),
      // "--border": "#FFFFFF"
    } as React.CSSProperties}>
      <div className='pr-5 pl-9 py-2 pb-1 rounded-t-md flex flex-row justify-between text-sm bg-input' style={{
        // color: (shikiTheme.backgroundColor || "#000000"), 
        // backgroundColor: (shikiTheme.textColor || "#FFFFFF"),
      }}>
        <p className='font-consolas h-8 text-center flex flex-col justify-center text-primary border-none'>{language.preview}</p>
        <Button className='m-0 h-8 text-primary hover:text-primary/75 active:text-primary/50 bg-transparent hover:bg-transparent active:bg-transparent' onClick={() => {
          handleCopy(text + unProcessedText);
        }}>
          <Copy className="w-4 h-4"/>
          <p className='pl-[9px]'>{"Copy"}</p>
        </Button>
      </div>
      <pre className="p-0 pt-1 flex flex-row rounded-lg text-sm ">
        <code className="pt-[0px] pb-[20px] pl-[7px] pr-[7px] !whitespace-pre select-none border-opacity-100 border-secondary">
          {Array(lineCount).fill(20).map((e, line_number: number) => (
            <span key={line_number}>
              {line_number + 1}
              {"\n"}
            </span>
          ))}
        </code>
        <ScrollAreaHorizontal className='min-w-auto'>
          {/* <pre>{text}</pre> */}
          {(codeHTML === "")?(
            <pre className='pl-[5px] pt-[0px] pb-[20px] pr-[8px]'>{text}</pre>
          ):(
            <div className='pl-[5px] pt-[0px] pb-[20px] pr-[8px]' dangerouslySetInnerHTML={{__html: codeHTML}}/>
          )}
        </ScrollAreaHorizontal>
      </pre>
    </div>
  );
}