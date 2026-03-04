"use client";
import { Dispatch, SetStateAction, useState, Fragment, useRef, memo } from "react"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import {
	divisionSection,
	headerSection,
	displaySection,
  contentSection,
  alignType
} from "@/types/toolchain-interface";
import { HeaderSection } from "./section-header";
import { ContentSection } from "./section-content";

const large_array = Array(350).fill(0);

export function DivisibleSection({
  onCollapse,
  onSectionUpdate,
  windowNumber,
  currentWindowCount,
  setCurrentWindowCount,
  sectionInfo
}:{
  onCollapse: () => void,
  onSectionUpdate: (section : displaySection) => void,
  windowNumber: number,
  currentWindowCount: number,
  setCurrentWindowCount: Dispatch<SetStateAction<number>>,
  sectionInfo: displaySection,
}) {
  const sizes = useRef<number[]>([]);
  const [section, setSection] = useState<displaySection>(JSON.parse(JSON.stringify(sectionInfo)));
  const sectionRef = useRef<displaySection>(JSON.parse(JSON.stringify(sectionInfo)));


  const updateSectionUpstream = (sectionLocal : displaySection) => {
    sectionRef.current = JSON.parse(JSON.stringify(sectionLocal));
    onSectionUpdate(sectionRef.current);
  }

  const updateSection = (sectionLocal : displaySection) => {
    setSection(sectionLocal);
    updateSectionUpstream(sectionLocal);
  }

  const onSplit = (splitType : "horizontal" | "vertical" | "header" | "footer", count: number) => {
    if (splitType == "horizontal" || splitType == "vertical") {

      let sections_array : contentSection[] = Array(count).fill({
        split: "none",
        size: Math.min(100/count, 100),
        tailwind: "",
        align: "center",
        mappings: []
      });

      sizes.current = Array(count).fill(Math.min(100/count, 100));
      
      sections_array[0] = JSON.parse(JSON.stringify(sectionRef.current)) as contentSection;
      sections_array[0].size = Math.min(100/count, 100);
      sections_array[0].header = undefined;
      sections_array[0].footer = undefined;

      const { mappings, ...sectionInfoReduced } = sectionRef.current as contentSection;

      const new_section : divisionSection = {
        ...sectionInfoReduced,
        split: splitType as "horizontal" | "vertical",
        sections: sections_array
      }
      // setSection(new_section);
      updateSection(new_section);
    } else if (splitType === "header") {
      const new_section : displaySection = {
        ...sectionRef.current,
        header: {
          align: "justify",
          tailwind: "",
          mappings: []
        }
      }
      // setSection(new_section);
      updateSection(new_section);
    } else if (splitType === "footer") {
      const new_section : displaySection = {
        ...sectionRef.current,
        footer: {
          align: "justify",
          tailwind: "",
          mappings: []
        }
      }
      // setSection(new_section);
      updateSection(new_section);
    }
  }

  const resetSection = (index : "header" | "footer" | number) => {
		if (index === "header") {
			const { header, ...new_section } = sectionRef.current;
			// setSection(new_section);
			updateSection(new_section);
		} 
		else if (index === "footer") {
			const { footer, ...new_section } = sectionRef.current;
			// setSection(new_section);
			updateSection(new_section);
		}
    else {

      if (sectionRef.current.split === "none") return;
			
      const { sections, ...new_section_one } = sectionRef.current;
			if (sections.length === 2) {
				const new_section : displaySection = {...new_section_one, ...sections[(index + 1) % 2]};
				// setSection(new_section);
				updateSection(new_section);
			} else {
				const new_section : divisionSection = {
					...new_section_one,
					sections: [
						...sections.slice(0, index),
						...sections.slice(index + 1)
					]
				}
				// setSection(new_section);
				updateSection(new_section);
			}
    }
  }

  const PrimaryContent = () => (
    <>
      {(section.split === "none" && (section as contentSection)) ? (
        <ContentSection
          onSplit={onSplit}
          onCollapse={onCollapse}
          onSectionUpdate={(sectionLocal) => {
            updateSectionUpstream(sectionLocal);
          }}
          sectionInfo={section as contentSection}
        />
      ):(
        <ResizablePanelGroup direction={(section as divisionSection).split} onLayout={(sizes: number[]) => {
          // Persist layout proportions in the section content.

          updateSectionUpstream({
            ...sectionRef.current as divisionSection, 
            sections: (sectionRef.current as divisionSection).sections.map(
              (sectionInner, index) => ({
                ...sectionInner, 
                size: sizes[index]
              })
            )
          })
        }}>
          {(section as divisionSection).sections.map((split_section, index) => (
            <Fragment key={index}>
              <ResizablePanel defaultSize={split_section.size}>
                <DivisibleSection
                  onCollapse={() => {resetSection(index)}}
                  onSectionUpdate={(sectionLocal) => {
                    //TODO Problem probably originates here.

                    // let new_section = sectionInfo as divisionSection;
                    // new_section.sections[index] = sectionLocal;
                    // setSection(new_section);
                    updateSectionUpstream({...(sectionRef.current as divisionSection), sections: [
                      ...(sectionRef.current as divisionSection).sections.slice(0, index),
                      sectionLocal,
                      ...(sectionRef.current as divisionSection).sections.slice(index+1)
                    ]});
                  }}
                  windowNumber={windowNumber}
                  currentWindowCount={currentWindowCount}
                  setCurrentWindowCount={setCurrentWindowCount}
                  sectionInfo={split_section}
                />
              </ResizablePanel>
              {(index < (section as divisionSection).sections.length - 1) && (
                <ResizableHandle/>
              )}
            </Fragment>
          ))}
        </ResizablePanelGroup>
      )}
    </>
  );


  return (
    <>
    {(section.header !== undefined || section.footer !== undefined) ? (
      <ResizablePanelGroup direction="vertical">
        {(section.header !== undefined) && (
          <HeaderSection 
            onCollapse={() => {resetSection("header")}} 
            onSectionUpdate={(s : headerSection) => {

              // TODO: If it breaks, it's probably this
              updateSectionUpstream({...section, header: s});
            }} 
            sectionInfo={section.header}
          />
        )}
        <ResizablePanel defaultSize={100}>
          <PrimaryContent />
        </ResizablePanel>
        {(section.footer !== undefined) && (
          <HeaderSection 
            onCollapse={() => {resetSection("footer")}} 
            onSectionUpdate={(s : headerSection) => {
              updateSectionUpstream({...section, footer: s});
            }} 
            sectionInfo={section.footer}
            type="footer"
          />
        )}
      </ResizablePanelGroup>
    ) : (
      <PrimaryContent />
    )}
    </>
  );
}

