"use client";

import React from "react"
import { textSegment } from "./markdown-text-splitter"
import { cn } from "@/lib/utils"
import { fontConsolas } from "@/lib/fonts"
import MarkdownLatex from "./markdown-latex"

export type segment_types = "regular" | "bold" | 
                            "italic" | "bolditalic" | 
                            "double_dollar" | "single_dollar" | 
                            "escaped_square_brackets" | "escaped_parentheses" |
                            "codespan" | "anchor" | 
                            "strikethrough" | "citation" | "code"

export type rendering_types = "regular" | "bold" | 
                              "italic" | "bolditalic" | 
                              "inline_math" | "newline_math" |
                              "codespan" | "anchor" | 
                              "strikethrough"


// export type markdownRenderingConfig = {[key in segment_types]: rendering_types}

// export type markdownRenderingConfig = Partial<{[key in segment_types]: rendering_types}>

export type markdownRenderingConfig = Partial<{
  [key in segment_types]: (textSeg: textSegment) => JSX.Element
}>


const textSegmentRenderers = {
  bold: (textSeg: textSegment) => (
    <strong className="font-bold">{textSeg.text}</strong>
  ),
  
  italic: (textSeg: textSegment) => (
    <em>{textSeg.text}</em>
  ),
  
  bolditalic: (textSeg: textSegment) => (
    <em><strong className="font-bold">{textSeg.text}</strong></em>
  ),
  
  regular: (textSeg: textSegment) => (
    <>{textSeg.raw_text}</>
  ),
  
  codespan: (textSeg: textSegment) => (
    <code className={cn("", fontConsolas.className)}>
      `{textSeg.text}`
    </code>
  ),
  
  strikethrough: (textSeg: textSegment) => (
    <del>{textSeg.text}</del>
  ),
  
  anchor: (textSeg: textSegment) => (
    <a href={textSeg.link as string} 
       target="_blank" 
       className="p-0 m-0 text-[#A68AEB] underline-offset-4 hover:underline active:text-[#A68AEB]/90">
      {textSeg.text}
    </a>
  ),
  
  newline_math: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="newline"/>
  ),
  
  inline_math: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="inline"/>
  ),
} as markdownRenderingConfig;

const normal_config: markdownRenderingConfig = {
  regular: textSegmentRenderers.regular,
  bold: textSegmentRenderers.bold,
  italic: textSegmentRenderers.italic,
  bolditalic: textSegmentRenderers.bolditalic,
  codespan: textSegmentRenderers.codespan,
  anchor: textSegmentRenderers.anchor,
  strikethrough: textSegmentRenderers.strikethrough,
}

export const OBSIDIAN_MARKDOWN_RENDERING_CONFIG: markdownRenderingConfig = {
  ...normal_config,
  single_dollar: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="inline"/>
  ),
  double_dollar: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="newline"/>
  ),
}

export const CHAT_RENDERING_STYLE: markdownRenderingConfig = {
  ...normal_config,
  double_dollar: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="newline"/>
  ),
  single_dollar: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="inline"/>
  ), // Ideally this is disabled
  escaped_parentheses: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="inline"/>
  ),
  escaped_square_brackets: (textSeg: textSegment) => (
    <MarkdownLatex textSeg={textSeg} type="newline"/>
  )
}



