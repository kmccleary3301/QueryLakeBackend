import { Metadata } from "next"

import "public/registry/themes.css"
import FlowDisplay from "./components/flow-page"
import LegacyNotice from "@/components/legacy/legacy-notice";

export const metadata: Metadata = {
  title: "Nodes",
  description: "Test XYFlow Nodes.",
}

export default function NodeEditorPage() {
  return (
    <div className="relative h-[calc(100vh)] w-full pr-0 pl-0">
      <div className="absolute left-4 right-4 top-4 z-50">
        <LegacyNotice
          title="Legacy toolchain builder"
          description="This is the legacy toolchain builder. Use the workspace Toolchains pages for the new workspace-first surface."
          workspacePath="/toolchains"
          ctaLabel="Open workspace Toolchains"
        />
      </div>
      <FlowDisplay/>
    </div>
  )
}
