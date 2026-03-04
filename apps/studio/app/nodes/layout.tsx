import { Metadata } from "next"
import { NodesNav } from "./nodes-nav"
import { NodeContextProvider } from "./context-provider"
import { useRef } from "react"

export const metadata: Metadata = {
  title: "Examples",
  description: "Check out some examples app built using the components.",
}

interface ExamplesLayoutProps {
  children: React.ReactNode
}

export default function NodesLayout({ children }: ExamplesLayoutProps) {
  
  return (
    <>
      <div>
        <NodeContextProvider interfaceConfiguration={{
          split: "none",
          size: 100,
          align: "center",
          tailwind: "",
          mappings: []
        }}>
        {/* <section>
          <div className="overflow-hidden rounded-[0.5rem] border bg-background shadow-md md:shadow-xl">
            
          </div>
        </section> */}
        <div className="absolute z-50 w-full flex flex-row justify-center pb-3 pointer-events-none">
          <div className="pointer-events-auto">
            <NodesNav/>
          </div>
        </div>
        {children}
        </NodeContextProvider>
      </div>
    </>
  )
}
