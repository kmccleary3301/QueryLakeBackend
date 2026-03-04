"use client";
import ScrollSection from "@/components/manual_components/scrollable-bottom-stick/custom-scroll-section";
import {
  contentSection,
  contentDiv,
  alignType,
	contentMapping
} from "@/types/toolchain-interface";
// import DisplayMappings from "@/components/toolchain-display-mappings-app";
import DisplayMappings from "@/public/cache/toolchains/toolchain-app-mappings";
import { useState, useRef, Fragment } from "react";
// import tailwindToStyle from "@/hooks/tailwind-to-obj/tailwind-to-style-obj";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";

export function ContentDiv({
  section = {type: "div", align: "center", tailwind: "min-w-[20px] min-h-[20px]", mappings: []}
}:{
  section: contentDiv,
}) {
	const { 
		breakpoint
  } = useContextAction();

  return (
    <div className={`flex flex-row justify-${
      (section.align === "justify") ? "around" :
      (section.align === "left") ?    "start"   :
      (section.align === "center") ?  "center"  :
      "end"
    }`}>
      <div className="" style={tailwindToObject([section.tailwind], breakpoint)}>
        {section.mappings.map((mapping, index) => (
          <Fragment key={index}>
            {((mapping as contentDiv).type && (mapping as contentDiv).type === "div") ? (
              <ContentDiv
                section={mapping as contentDiv}
              />
            ) : (
              <DisplayMappings
                info={mapping as contentMapping}
              />

            )}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
