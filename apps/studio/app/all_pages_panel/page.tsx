"use client";
import "public/registry/themes.css";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { motion } from 'framer-motion';
import { ModeToggle } from "@/components/inherited/mode-toggle"
import { ScrollArea } from "@/components/ui/scroll-area";

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15
    }
  }
};

const childVariants = {
  hidden: { opacity: 1, scale: 1.2, filter: "blur(5px)" },
  show: { opacity: 1, scale: 1, filter: "blur(0px)", transition: { duration: 0.4} },
};

const buttonProps : {header : string, content : string, link : string}[] = [
  {
    header: "Collection",
    content: "A bad URL for the collections page",
    link: "/collection/12038/19192?a=312"
  },
  {
    header: "Themes",
    content: "Original ShadCN website theme page",
    link: "/themes"
  },
  {
    header: "Temes",
    content: "Exact copy of the original ShadCN website theme page",
    link: "/temes"
  },
  {
    header: "Toolchain Editor",
    content: "The toolchain editor page, AKA nodes",
    link: "/nodes"
  },
  {
    header: "Aceternity",
    content: "Aceternity examples",
    link: "/aceternity"
  },
  {
    header: "Examples",
    content: "Original ShadCN website examples page",
    link: "/themes"
  },
  {
    header: "Auth",
    content: "Authentication page",
    link: "/auth"
  },
  {
    header: "App",
    content: "QueryLake application interface",
    link: "/app"
  },
  {
    header: "Home",
    content: "The home page",
    link: "/home"
  },
  {
    header: "Test",
    content: "Test components",
    link: "/test"
  },
  {
    header: "Test V2",
    content: "Test more components",
    link: "/test/v2"
  },
  {
    header: "InfiniTable",
    content: "Test new infinite table",
    link: "/test/new_table"
  }
]


export default function AllPagesPanelPage() {
  return (
    <div className="w-full h-[calc(100vh)] flex flex-col justify-center">
      <ScrollArea className="w-full">
      <div className="w-full flex flex-row justify-center">
        <motion.div className="flex-shrink flex flex-wrap justify-center gap-4 w-[80vw]"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >

          {buttonProps.map((button, index) => (
            <motion.div className="h-[120px] w-[250px] " variants={childVariants} key={index}>
              <Link href={button.link}>
                <Button variant="ghost" className="h-full w-full rounded-lg overflow-auto whitespace-normal items-center py-2 px-4">
                  <div className="h-full w-full flex flex-col justify-center lg:justify-start space-y-3">
                    <p className="w-[90%] text-base lg:text-lg text-left"><strong>{button.header}</strong></p>
                    <div className="bg-secondary rounded-full w-[90%] h-[2px]"/>
                    <p className="w-[90%] text-xs lg:text-sm break-words text-left">{button.content}</p>
                  </div>
                </Button>
              </Link>
            </motion.div>
          ))}
          {/* <motion.div className="h-[160px] w-[300px] rounded-lg" variants={childVariants}>
              <div className="h-full w-full flex flex-col justify-center lg:justify-start space-y-3">
                <p className="w-[90%] text-base lg:text-lg text-left"><strong>{"Toggle Display Mode"}</strong></p>
                <div className="bg-secondary rounded-full w-[90%] h-[2px]"/>
                <ModeToggle/>
              </div>
          </motion.div> */}
        </motion.div>
      </div>
      </ScrollArea>
    </div>
  )
}