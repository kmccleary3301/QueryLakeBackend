"use client";
import {
  contentDiv,
	contentMapping,
	headerSection
} from "@/types/toolchain-interface";
// import DisplayMappings from "@/components/toolchain-display-mappings-app";
import DisplayMappings from "@/public/cache/toolchains/toolchain-app-mappings";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import { Fragment } from "react";
import { ContentDiv } from "./section-div";

export function HeaderSection({
	section = {align: "justify", tailwind: "", mappings: []},
	type = "header"
}:{
	section: headerSection,
	type?: "header" | "footer"
}) {

	const { 
		breakpoint
  } = useContextAction();

	return (
		<>
		{(typeof section.tailwind === "string") &&
		<div className={`text-center flex flex-row`}>
			
				<div className={`w-full h-full flex flex-row justify-${
					(section.align === "justify") ? "around" :
					(section.align === "left") ?    "start"   :
					(section.align === "center") ?  "center"  :
					"end"
				}`} style={tailwindToObject([section.tailwind], breakpoint)}>
					{(section.mappings.length === 0) && (<p className="select-none">{(type === "header")?"Header":"Footer"}</p>)}

					{section.mappings.map((mapping, index) => (
							// <div key={index} className="h-[50px]">{mapping.display_as}</div>
							// <DisplayMappings
							// 	key={index}
              //   info={mapping}
							// />
              <Fragment key={index}>
                {((mapping as contentDiv).type && (mapping as contentDiv).type === "div") ? (
                  <ContentDiv section={mapping as contentDiv}/>
                ) : (
                  <DisplayMappings info={mapping as contentMapping}/>
                )}
              </Fragment>
						))}
				</div>
		</div>}
		</>
	)
}