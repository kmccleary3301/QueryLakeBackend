"use client";
import { usePathname } from "next/navigation";
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { useContextAction } from "@/app/context-provider";
import { motion, useAnimation } from "framer-motion";
import * as Icon from 'react-feather';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import { Home, Settings, Sidebar } from "lucide-react";

export default function SidebarTemplate({ 
	children,
	width = "320px",
  className = "",
  buttonsClassName = ""
}:{ 
	children: React.ReactNode,
	width?: string | number,
  className?: string,
  buttonsClassName?: string,
}) {
  const pathname = usePathname();
  const { 
    userData,
    sidebarOpen,
    setSidebarOpen,
  } = useContextAction();


  const width_as_string = (typeof width === 'string') ? `[${width}]` : width.toString();

  const controlsSidebarWidth = useAnimation();
  const [sidebarToggleVisible, setSidebarToggleVisible] = useState<boolean>(true);
  
  const controlSidebarButtonOffset = useAnimation();

  useEffect(() => {
		controlsSidebarWidth.start({
			width: (sidebarOpen)?width:0
		});
	}, [controlsSidebarWidth, sidebarOpen, width]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setSidebarToggleVisible(!sidebarOpen);
      controlSidebarButtonOffset.set({ zIndex: sidebarOpen?0:52 });
    }, sidebarOpen?0.4:0);

    controlsSidebarWidth.start({
      width: (sidebarOpen)?width:0,
      transition: { duration: 0.4, ease: "easeInOut", bounce: 0 }
    });

    return () => {
      clearTimeout(timeoutId);
    };
	}, [sidebarOpen, width, controlsSidebarWidth, controlSidebarButtonOffset]);

  useEffect(() => {
    // const sidebar_value = (sidebarIsAvailable && sidebarOpened)?true:false;

    controlSidebarButtonOffset.start({
			translateX: sidebarOpen?0:0,
      opacity: sidebarOpen?0:1,
			transition: { delay: sidebarOpen?0:0.4, duration: sidebarOpen?0:0.6, ease: "easeInOut", bounce: 0 }
		});
  }, [sidebarOpen, controlSidebarButtonOffset, pathname]);


	return (

    <>
      <motion.div 
        id="SIDEBARBUTTON" 
        className={`p-1 pl-2 absolute`}
        initial={{translateX: sidebarOpen?0:0, opacity: sidebarOpen?0:1, zIndex: 52}}
        animate={controlSidebarButtonOffset}>
        {(sidebarToggleVisible) ? (
          <Button variant="ghost" className={`p-2 rounded-md pl-2 pr-2 text-primary active:text-primary/70`} onClick={() => {
            setSidebarOpen(true);
          }}>
            <Sidebar id="closed_sidebar_button" size={24}/>
          </Button> 
        ):null}
      </motion.div>
      
      <div className="h-screen">
        <motion.div className="h-full bg-background-sidebar flex flex-col p-0 z-54" initial={{width: (sidebarOpen)?width:0}} animate={controlsSidebarWidth} >
          {(userData === undefined) ? (
            null
          ) : (
          <div className='w-full h-full border-accent'>
            <div className={cn('max-h-screen h-full flex flex-col', `w-${width_as_string}`, className)}>
              {/* <div className="flex-grow px-0 flex flex-col"> */}
                <div className={cn("flex flex-row pt-1 pb-[7.5px] px-30 items-center w-full justify-between", buttonsClassName)}>
                  <Link href="/home">
                    <Button variant="ghost" className="p-2 rounded-md pl-2 pr-2 text-primary active:text-primary/70">
                      <Home size={24}/>
                    </Button>
                  </Link>
                  <Link href="/settings">
                    <Button variant="ghost" className="p-2 rounded-md pl-2 pr-2 text-primary active:text-primary/70">
                      <Settings size={24}/>
                    </Button>
                  </Link>
                  <Button variant="ghost" className="p-2 rounded-md pl-2 pr-2 text-primary active:text-primary/70" onClick={() => {
                    // TODO: Toggle Sidebar
                    setSidebarOpen(false);
                  }}>
                    {/* <Icon.Sidebar size={24} color="#E8E3E3" /> */}
                    <Sidebar size={24}/>
                  </Button>
                </div>
                <div className='w-full h-[calc(100vh-52px)] flex flex-col'>
                  {children}
                </div>
              {/* </div> */}
            </div>
          </div>
          )}
        </motion.div>
      </div>
    </>
	);
}
