"use client";
import * as Icon from 'react-feather';
import HoverDocumentEntry from '../manual_components/hover_document_entry';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { userDataType, timeWindowType, collectionGroup, toolchain_session, setStateOrCallback } from '@/types/globalTypes';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useContextAction } from "@/app/context-provider";
import { Trash } from 'lucide-react';
import { cn } from '@/lib/utils';
import { HoverTextDiv } from '@/components/ui/hover-text-div';
import SidebarEntry from '../manual_components/sidebar-entry-fade';


function SessionEntry({
  session,
  selected = false,
  onDelete = () => {},
}:{
  session: toolchain_session,
  selected?: boolean,
  onDelete?: () => void,
}) {
  return (
    <>
      {selected ? (
        <div className={cn(
          "relative bg-secondary text-secondary-foreground hover:bg-accent active:bg-secondary/60",
          'p-0 w-full flex flex-row-reverse justify-between h-8 rounded-lg'
        )}>
          <div className='w-full text-left flex flex-col justify-center rounded-[inherit]'>
            <p className='relative px-2 overflow-hidden text-sm whitespace-nowrap'>{session.title}</p>
          </div>
          <Link href={`/app/session?s=${session.id}`} className="absolute w-[40px] h-8 rounded-r-[inherit] bg-gradient-to-l from-accent to-accent/0"/>
          <div className='h-8 absolute flex flex-col justify-center opacity-0 hover:opacity-100 rounded-r-[inherit]'>
            <div className='h-auto flex flex-row rounded-r-[inherit]'>
              <Link href={`/app/session?s=${session.id}`} className="w-[40px] h-auto rounded-md bg-gradient-to-l from-accent to-accent/0"/>
              <div className="bg-accent rounded-r-[inherit] display-none">
              <Button className='h-6 w-6 rounded-full p-0 m-0' variant={"ghost"} onClick={onDelete}>
                <Trash className='w-3.5 h-3.5 text-primary'/>
              </Button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className={cn(
          "relative hover:bg-accent active:bg-accent/70 hover:text-accent-foreground hover:text-accent-foreground/",
          'p-0 w-full flex flex-row-reverse justify-between h-8 rounded-lg'
        )}>
          <HoverTextDiv hint={session.title} className='w-full text-left flex flex-col justify-center rounded-[inherit]'>
            <Link href={`/app/session?s=${session.id}`} className='rounded-[inherit] w-full'>
              <p className='relative px-2 overflow-hidden overflow-ellipsis text-sm whitespace-nowrap'>{session.title}</p>
            </Link>
          </HoverTextDiv>
          {/* <div className="h-8 absolute w-[50px] bg-indigo-500"/> */}
          <div className='h-8 absolute flex flex-col justify-center opacity-0 hover:opacity-100 rounded-r-[inherit]'>
            <div className='h-auto flex flex-row rounded-r-[inherit]'>
              <Link href={`/app/session?s=${session.id}`} className="w-[40px] h-auto rounded-md bg-gradient-to-l from-accent to-accent/0"/>
              <div className="bg-accent rounded-r-[inherit] display-none">
              <Button className='h-6 w-6 rounded-full p-0 m-0' variant={"ghost"} onClick={onDelete}>
                <Trash className='w-3.5 h-3.5 text-primary'/>
              </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}


export default function SidebarChatHistory({
  scrollClassName,
}:{
  scrollClassName : string,
}) {
  const {
    toolchainSessions,
    setToolchainSessions,
    activeToolchainSession,
    setActiveToolchainSession,
  } = useContextAction();

  const router = useRouter();
  const [internalToolchainSessions, setInternalToolchainSessions] = useState<timeWindowType[]>([]);

  useEffect(() => {

    const timeWindows : timeWindowType[] = [
			{title: "Last 24 Hours", cutoff: 24*3600, entries: []},
			{title: "Last 2 Days", cutoff: 2*24*3600, entries: []},
			{title: "Past Week", cutoff: 7*24*3600, entries: []},
			{title: "Past Month", cutoff: 30*24*3600, entries: []},
      {title: "Past Year", cutoff: 365*24*3600, entries: []},
      {title: "Older", cutoff: Infinity, entries: []}
		];


    const current_time = Math.floor(Date.now() / 1000);

    // const newToolchainSessions = new Map<string, toolchain_session>();

    toolchainSessions.forEach((session : toolchain_session) => {
      const delta_time = current_time - session.time;
      for (let i = 0; i < timeWindows.length; i++) {
        if (delta_time < timeWindows[i].cutoff) {
          timeWindows[i].entries.push(session);
          break;
        }
      }
    });

    for (let i = 0; i < timeWindows.length; i++) {
      timeWindows[i].entries.sort((a : toolchain_session, b : toolchain_session) => (b.time - a.time));
    }

    // console.log("Time Windows:", timeWindows);

    setInternalToolchainSessions(timeWindows);
  }, [toolchainSessions])


  const deleteSession = (hash_id : string) => {
    // TODO : Implement delete session with API
    
    // setToolchainSessions(toolchainSessions);
    // toolchainSessions.delete(hash_id);
    setToolchainSessions((prevSessions) => {
      // Create a new Map from the previous one
      const newMap = new Map(prevSessions);
      // Update the new Map
      newMap.delete(hash_id);
      // Return the new Map to update the state
      return newMap;
    });
    // setToolchainSessions(toolchainSessions.filter((session : toolchain_session) => (session.id !== hash_id)));
  };
  
  return (
    <div className='pb-0 overflow-hidden'>
      <div className='pb-0 pt-2'>
        <Link href="/app/create">
          <Button variant={"ghost"} className="w-full flex flex-row rounded-2xl h-9 items-center justify-center">
              <div style={{paddingRight: 5}}>
                <Icon.Plus size={20}/>
              </div>
              <div style={{alignSelf: 'center', justifyContent: 'center'}}>
              <p>{"New Session"}</p>
              </div>
          </Button>
        </Link>
      </div>
      <ScrollArea className={cn("pb-0 -mr-4 pt-2", scrollClassName)}>
        <div className='space-y-6 pr-4'>
        {internalToolchainSessions.map((chat_history_window : timeWindowType, chat_history_index : number) => (
          <div key={chat_history_index}>
            {(chat_history_window.entries.length > 0) && (
              <div className='space-y-1 w-[220px]'>
                <p className="w-full text-left text-sm text-primary/50">
                  {chat_history_window.title}
                </p>
                {chat_history_window.entries.map((value : toolchain_session, index : number) => (
                  // <SessionEntry key={index} session={value} selected={(value.id === activeToolchainSession)}/>
                  <SidebarEntry key={index} href={`/app/session?s=${value.id}`} title={value.title} className='pl-4' displaySelected={false}>
                    <span className="flex flex-row justify-center pointer-events-auto gap-x-2 pr-2">
                      <button onClick={() => {
                        deleteSession(value.id);
                      }} className='h-6 w-4 rounded-full p-0 m-0 text-primary active:text-primary/70'>
                        <Trash className='w-3.5 h-3.5'/>
                      </button>
                    </span>
                  </SidebarEntry>
                ))}
              </div>
            )}
          </div>
        ))}
       </div>
      </ScrollArea>
    </div> 
  );
}