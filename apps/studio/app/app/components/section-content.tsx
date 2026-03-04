"use client";
import ScrollSection from "@/components/manual_components/scrollable-bottom-stick/custom-scroll-section";
import {
  contentDiv,
  contentMapping,
  contentSection
} from "@/types/toolchain-interface";
// import DisplayMappings from "@/components/toolchain-display-mappings-app";
import DisplayMappings from "@/public/cache/toolchains/toolchain-app-mappings";
// import tailwindToStyle from "@/hooks/tailwind-to-obj/tailwind-to-style-obj";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import { Fragment } from "react";
import { ContentDiv } from "./section-div";

export function ContentSection({
  section
}:{
  section: contentSection,
}) {
	const { 
		breakpoint
  } = useContextAction();

  return (
    <div className="w-full h-full">
			<ScrollSection scrollBar={true} scrollToBottomButton={true} innerClassName={`w-full`}>
				<div className={`flex flex-row justify-${
					(section.align === "justify") ? "around" :
					(section.align === "left") ?    "start"   :
					(section.align === "center") ?  "center"  :
					"end"
				}`}>
					<div style={
						tailwindToObject(["flex flex-col", section.tailwind], breakpoint)
					}>
						{section.mappings.map((mapping, index) => (
              <Fragment key={index}>
                {((mapping as contentDiv).type && (mapping as contentDiv).type === "div") ? (
                  <ContentDiv
                    section={mapping as contentDiv}
                  />
                ) : (
                  <DisplayMappings
                    info={(mapping) as contentMapping}
                  />
                )}
              </Fragment>
						))}
					</div>
				</div>
			</ScrollSection>
    </div>
  );
}

