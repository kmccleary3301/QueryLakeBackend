import { ScrollArea } from "@/components/ui/scroll-area"

export default function DocsLayout({ 
	children 
} : {
	children: React.ReactNode
}) {
  
	return (
		<div className="w-full h-[calc(100vh)] absolute">
      <div className="w-full h-[calc(100vh)]">
        <ScrollArea className="w-full h-screen @container/docs">
          <div className="flex flex-row justify-center">
            <div className="md:w-[min(90cqw,65vw)] lg:w-[min(90cqw,55vw)] xl:w-[min(90cqw,45vw)] w-[min(90cqw,80vw)]">
              <div className="px-[3rem]">
								{children}
							</div>
              <div className="h-[100px]"/>
            </div>
          </div>
        </ScrollArea>
      </div>
		</div>
	)
}