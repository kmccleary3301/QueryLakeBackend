"use client";
import { Fragment, useEffect, useRef, useState } from "react"
import {
  ContextMenuHeaderWrapper
} from "./context-menu-wrapper";
import {
	alignType,
	contentDiv,
	contentMapping,
	contentSection,
	headerSection
} from "@/types/toolchain-interface";
import { cn } from "@/lib/utils";
// import DisplayMappings from "@/components/toolchain-display-mappings-editor";
import DisplayMappings from "@/public/cache/toolchains/toolchain-editor-mappings";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import { ContentDiv } from "./section-div";

export function HeaderSection({
	onCollapse,
	onSectionUpdate,
	sectionInfo = {align: "justify", tailwind: "", mappings: []},
	type = "header"
}:{
	onCollapse: () => void,
	onSectionUpdate: (section : headerSection) => void,
	sectionInfo: headerSection,
	type?: "header" | "footer"
}) {

	const { 
		breakpoint
  } = useContextAction();

	const [section, setSection] = useState<headerSection>(sectionInfo);
	const sectionRef = useRef<headerSection>(sectionInfo);

	const updateSectionUpstream = (sectionLocal : headerSection) => {
    sectionRef.current = JSON.parse(JSON.stringify(sectionLocal));
    onSectionUpdate(sectionRef.current);
  }

  const updateSection = (sectionLocal : headerSection) => {
    setSection(sectionLocal);
    updateSectionUpstream(sectionLocal);
  }


	useEffect(() => {
		console.log(sectionInfo);
		// console.log("New Header Tailwind:", sectionInfo.tailwind);
	}, [sectionInfo]);

	return (
		<>
		{(typeof sectionInfo.tailwind === "string") &&
		<div className={`text-center flex flex-row`}>
			<ContextMenuHeaderWrapper 
        className="h-full w-full"
				onCollapse={onCollapse} 
				onAlign={(a : alignType) => {
					updateSection({...section, align: a})
				}}
				setTailwind={(t : string) => {
					updateSection({...section, tailwind: t} as headerSection)
				}}
				addComponent={(component) => {
					updateSection({...section, mappings: [...section.mappings, component]} as headerSection);
				}}
				align={section.align}
				tailwind={section.tailwind}
			>
				<div className={`w-full h-full border-primary/50 border-[2px] border-dashed flex flex-row justify-${
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
							// 	info={mapping}
							// 	onDelete={() => {updateSection({
							// 		...section, 
							// 		mappings: [...section.mappings.slice(0, index), ...section.mappings.slice(index+1)]
							// 	} as contentSection)}}
							// 	setInfo={(value : contentMapping) => {
							// 		updateSection({
							// 			...section, 
							// 			mappings: [...section.mappings.slice(0, index), value, ...section.mappings.slice(index+1)]
							// 		} as contentSection);
							// 	}}
							// />
              <Fragment key={index}>
                {((mapping as contentDiv).type && (mapping as contentDiv).type === "div") ? (
                  <ContentDiv
                    onCollapse={() => {updateSection({
                      ...section, 
                      mappings: [...section.mappings.slice(0, index), ...section.mappings.slice(index+1)]
                    } as contentSection)}}
                    onSectionUpdate={(value : contentDiv) => {
                      updateSection({
                        ...section, 
                        mappings: [...section.mappings.slice(0, index), value, ...section.mappings.slice(index+1)]
                      } as contentSection);
                    }}
                    sectionInfo={mapping as contentDiv}
                  />
                ) : (
                  <DisplayMappings
                    info={mapping as contentMapping}
                    onDelete={() => {updateSection({
                      ...section, 
                      mappings: [...section.mappings.slice(0, index), ...section.mappings.slice(index+1)]
                    } as contentSection)}}
                    setInfo={(value : contentMapping) => {
                      updateSection({
                        ...section, 
                        mappings: [...section.mappings.slice(0, index), value, ...section.mappings.slice(index+1)]
                      } as contentSection);
                    }}
                  />
                )}
              </Fragment>
						))}
				</div>
			</ContextMenuHeaderWrapper>
		</div>}
		</>
	)
}
