"use client";

import MarkdownRenderer from "@/components/markdown/markdown-renderer";
import allDocs from "@/public/cache/documentation/__all-documents__";
import { getValueFromPath } from "./hooks";
import React, { Usable, use } from "react";
import { OBSIDIAN_MARKDOWN_RENDERING_CONFIG } from "@/components/markdown/configs";
import { useParams } from "next/navigation";

export default function DocPage() {
  const resolvedParams = useParams() as {
    slug: string[],
  };
  
  const { slug } = resolvedParams;
  const doc : { slug : string, content : string } = getValueFromPath(allDocs, slug);



  if (doc === undefined) {
    return (
      <>
        <div className='w-full h-[calc(100vh)] flex flex-col justify-center'>
          <h1>Doc Not Found</h1>
          <p className='w-full text-base text-primary/80 break-words text-left'>
            The doc you are looking for does not exist.  
          </p>
        </div>
      </>
    );
  }

  return (
    <div>
      <p className="text-5xl text-primary/80 text-bold pt-5 pb-10"><strong>{doc.slug}</strong></p>
      {/* <p className="text-lg text-primary text-bold pb-10 pt-5">
        <em>By Kyle McCleary</em>
      </p> */}
      <MarkdownRenderer input={doc.content} finished={true} config={OBSIDIAN_MARKDOWN_RENDERING_CONFIG}/>
    </div>
  );
}