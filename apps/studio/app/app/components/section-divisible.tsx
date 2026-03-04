"use client";
import { Fragment, useEffect} from "react"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import {
	divisionSection,
	displaySection,
  contentSection,
} from "@/types/toolchain-interface";
import { HeaderSection } from "./section-header";
import { ContentSection } from "./section-content";

export function PrimaryContent({ section }: { section: displaySection }) {

  return (
    <>
      {section.split === "none" && (section as contentSection) ? (
        <ContentSection section={section as contentSection} />
      ) : (
        <ResizablePanelGroup direction={(section as divisionSection).split}>
          {(section as divisionSection).sections.map((split_section, index) => (
            <Fragment key={index}>
              <ResizablePanel defaultSize={split_section.size}>
                <DivisibleSection section={split_section} />
              </ResizablePanel>
              {index < (section as divisionSection).sections.length - 1 && (
                <ResizableHandle />
              )}
            </Fragment>
          ))}
        </ResizablePanelGroup>
      )}
    </>
  );
};

export function DivisibleSection({
  section
}:{
  section: displaySection,
}) {

  return (
    <>
    {(section.header || section.footer) ? (
      <ResizablePanelGroup direction="vertical">
        {(section.header) && (
          <HeaderSection
            section={section.header}
          />
        )}
        <ResizablePanel defaultSize={100}>
          <PrimaryContent section={section} />
        </ResizablePanel>
        {(section.footer) && (
          <HeaderSection
            section={section.footer}
            type="footer"
          />
        )}
      </ResizablePanelGroup>
    ) : (
      <PrimaryContent section={section} />
    )}
    </>
  );
}

