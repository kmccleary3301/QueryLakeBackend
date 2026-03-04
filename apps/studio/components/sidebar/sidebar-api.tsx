"use client";

import { usePathname } from "next/navigation";
import SidebarTemplate from './sidebar-template';
import { 
  folder_structure, 
  folder_structure_aliases, 
  reverse_lookup 
} from "@/public/cache/documentation/__all-documents__";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fontSans } from "@/lib/fonts";
import { cn } from "@/lib/utils";
import Link from 'next/link';
import { Button } from "@/components/ui/button";
import { toVariableName } from "@/app/docs/[[...slug]]/hooks";
import { useState, useEffect } from 'react';
import { motion, useAnimation } from "framer-motion";
import { BarChart2, Database, Lock } from "lucide-react";

export function FolderViewer({ 
	content,
	route = [],
	topLevel = true,
}:{
	content: object,
	route?: string[],
	topLevel?: boolean
}) {
	const [isOpen, setIsOpen] = useState<boolean>((route.length === 0));

	const controlHeight = useAnimation();

	useEffect(() => {
		controlHeight.set({
			height: (route.length === 0)?"auto":0
		});
	}, [controlHeight, route.length]);

	useEffect(() => {
		controlHeight.start({
			height: (isOpen)?"auto":0,
			transition: { duration: 0.4 }
		});
  }, [isOpen, controlHeight]);


	return (
		<div className={`space-y-1`}>
			{(!topLevel) && (
				<Button variant={"ghost"} className="whitespace-nowrap h-9 px-2" onClick={() => {
					setIsOpen(!isOpen);
				}}>
					<p className={`font-bold whitespace-nowrap text-ellipsis`}>
						<strong>{route[route.length - 1]}</strong>
					</p>
				</Button>
			)}
			<motion.div 
				id="content-list" 
				className="text-sm antialiased w-full pl-4 flex flex-col space-y-1 whitespace-nowrap overflow-hidden"
				animate={controlHeight} 
			>
				<div className="overflow-hidden">
				{/* <div className="flex flex-col scrollbar-hide overflow-y-auto"> */}
					{Object.entries(content).map(([key, value]) => (
						<div key={key}>
							{(value === null)?(
								<Link 
									href={`/docs/${route.map((s : string) => toVariableName(s)).join("/")}/${toVariableName(key)}`} 
									className="flex items-center space-x-2"
								>
									<Button variant={"ghost"} className="whitespace-nowrap h-9 px-2">
										{key}
									</Button>
								</Link>
							):(
								<FolderViewer key={key} content={value} topLevel={false} route={[...route, key]} />
							)}
						</div>
					))}
				</div>
			</motion.div>
		</div>
	);
}

export default function ApiSidebar() {
  const pathname = usePathname() || "";
	return (
    <SidebarTemplate width={"220px"} className='px-2'>
      <div className={cn("flex flex-col w-full h-full", fontSans.className)}>
				<div className="flex flex-col w-full h-full">
					<ScrollArea className="pl-2 pr-2">
						<div className="pb-[15px] flex flex-col gap-1">
						{/* <FolderViewer content={folder_structure}/> */}
            
              <Link href="/platform/storage">
                <Button variant={(pathname.startsWith("/platform/storage"))?"secondary":"ghost"} className="w-full whitespace-nowrap h-9 px-2 text-primary active:text-primary/70">
                  <div className="w-full flex flex-row justify-start">
                    <Database className="w-4 h-4 my-auto mr-2"/>
                    <p className="h-auto flex flex-col justify-center">Storage</p>
                  </div>
                </Button>
              </Link>
              <Link href="/platform/usage">
                <Button variant={(pathname.startsWith("/platform/usage"))?"secondary":"ghost"} className="w-full items-start h-9 px-2 text-primary active:text-primary/70">
                  <div className="w-full flex flex-row justify-start">
                    <BarChart2 className="w-4 h-4 mr-2"/>
                    <p>Usage</p>
                  </div>
                </Button>
              </Link>
              <Link href="/platform/api">
                <Button variant={(pathname.startsWith("/platform/api"))?"secondary":"ghost"} className="w-full whitespace-nowrap h-9 px-2 text-primary active:text-primary/70">
                  <div className="w-full flex flex-row justify-start">
                    <Lock className="w-4 h-4 mr-2"/>
                    <p>API Keys</p>
                  </div>
                </Button>
              </Link>
						</div>
					</ScrollArea>
				</div>
			</div>
    </SidebarTemplate>
	);
}
