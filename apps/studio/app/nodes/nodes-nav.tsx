"use client"

import Link from "next/link"
import { useSearchParams } from 'next/navigation'
import { usePathname } from "next/navigation"
import { useState, useRef, useEffect } from "react"
import {
  Tabs,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { motion } from 'framer-motion';
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"
import { useNodeContextAction } from "./context-provider"


const examples = [
  {
    name: "Node Editor",
    href: "/nodes/node_editor"
  },
  {
    name: "Display Editor",
    href: "/nodes/display_editor"
  }
]

interface ExamplesNavProps extends React.HTMLAttributes<HTMLDivElement> {}

export function NodesNav({ className, ...props }: ExamplesNavProps) {

  const { 
    interfaceConfiguration, 
    // setInterfaceConfiguration
    getInterfaceConfiguration
  } = useNodeContextAction();

  const pathname = usePathname() || "";
  
  const searchParamsString = useSearchParams()?.toString() || undefined;
  const linkAppend = searchParamsString ? `?${searchParamsString}` : "";

  const [hover, setHover] = useState(false);

  const downloadHook = (json : object, filename : string) => {
    const data = JSON.stringify(json, null, '\t');
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="">
      <motion.div
        className="overflow-hidden bg-background-sidebar rounded-b-md"
        initial={{ height: 20 }}
        animate={ (hover) ? 
          { height: 'auto'} : 
          { height: 10}
        }
        transition={{ duration: 0.2 }}
        onHoverStart={() => setHover(true)}
        onHoverEnd={() => setHover(false)}
      >
        <div className="pb-2 pt-2 flex flex-col">
            <Tabs className="bg-background-sidebar" value={
              pathname.startsWith("/nodes/node_editor") ? "/nodes/node_editor" : 
              pathname.startsWith("/nodes/display_editor") ? "/nodes/display_editor" :
              undefined
            } onValueChange={(value : string) => {console.log("Value changed to", value)}}>
              <TabsList className="grid w-full grid-cols-2 rounded-none bg-background-sidebar">
                {examples.map((example, index) => (
                  <TabsTrigger className="data-[state=active]:bg-accent" key={index} value={example.href}>
                    <Link href={example.href+linkAppend}>
                      {example.name}
                    </Link>
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
            <div className="w-auto flex flex-row justify-end">
              <Button size="icon" variant="ghost" onClick={() => {
                downloadHook(getInterfaceConfiguration(), "interface-configuration.json");
              }}>
                <Download className="w-4 h-4 text-primary" />
							</Button>
            </div>
        </div>
      </motion.div>
    </div>
  );
}