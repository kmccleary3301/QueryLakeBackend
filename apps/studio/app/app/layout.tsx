import { ToolchainContextProvider } from "./context-provider"

interface DocsLayoutProps {
  children: React.ReactNode
}

export default function DocsLayout({ children }: DocsLayoutProps) {
  return (
    <div className="w-full h-full bg-background">
      <ToolchainContextProvider>
      {/* <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10"> */}
        {children}
      {/* </div> */}
      </ToolchainContextProvider>
    </div>
  )
}
