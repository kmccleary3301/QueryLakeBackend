"use client";
import { cn } from "@/lib/utils";
import MarkdownLatex from "./markdown-latex";
import { fontConsolas } from '@/lib/fonts';
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { textSegment } from "./markdown-text-splitter";
import { CHAT_RENDERING_STYLE, markdownRenderingConfig, OBSIDIAN_MARKDOWN_RENDERING_CONFIG } from "./configs";
import { useEffect } from "react";
import { object } from "zod";

export default function MarkdownTextAtomic({
  textSeg,
  config
}:{
  textSeg: textSegment,
  config: markdownRenderingConfig
}){

  // const config_map = config || OBSIDIAN_MARKDOWN_RENDERING_CONFIG;

  // const text_segment_lookup = config_map[textSeg.type];
  const renderer = config[textSeg.type];
  return renderer ? renderer(textSeg) : (
    <>{textSeg.raw_text}</>
  );
}