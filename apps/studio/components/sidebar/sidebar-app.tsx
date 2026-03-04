"use client";
import { useState, useEffect } from 'react';
import { getUserCollections } from '@/hooks/querylakeAPI';
import { 
  collectionGroup
} from '@/types/globalTypes';
import { Button } from '@/components/ui/button';
import {
	TabsContent,
  TabsList,
  TabsTrigger,
	Tabs
} from "@/components/ui/tabs"
import SidebarCollectionSelect from './sidebar-collections';
import SidebarToolchains from './sidebar-toolchains';
import SidebarChatHistory from './sidebar-history';
import Link from 'next/link';
import { useContextAction } from "@/app/context-provider";
import { deleteCookie } from '@/hooks/cookies';
import * as Icon from 'react-feather';
import SidebarTemplate from './sidebar-template';

export default function AppSidebar() {
  
  const { 
    userData, 
    setUserData, 
    collectionGroups,
    setCollectionGroups,
    refreshCollectionGroups,
    selectedCollections, 
    setSelectedCollections,
    selectedToolchain,
    setSelectedToolchain,
    authReviewed,
    loginValid
  } = useContextAction();

  // Get all user data for sidebar. This includes chat history and collections
  useEffect(() => {
		if (userData === undefined || !authReviewed || !loginValid) return;
    if (collectionGroups.length == 0) {
      refreshCollectionGroups();
    }
  }, [
    userData,
    authReviewed,
    loginValid,
    collectionGroups.length,
    refreshCollectionGroups
  ]);

  useEffect(() => {
    const new_selections_state = new Map<string, boolean>([]);
    for (let i = 0; i < collectionGroups.length; i++) {
      for (let j = 0; j < collectionGroups[i].collections.length; j++) {
        new_selections_state.set(collectionGroups[i].collections[j]["hash_id"], false); //Change to false
      }
    }
    setSelectedCollections(new_selections_state);
  }, [collectionGroups, setSelectedCollections]);

	const setCollectionSelected = (collection_hash_id : string, value : boolean) => {
		setSelectedCollections(selectedCollections.set(collection_hash_id, value));
  };

	return (
    <SidebarTemplate width={"260px"} className='pl-4 pr-4'>
      {(userData !== undefined) && (
        <>
          <div className="w-full px-22 pb-0 flex-grow flex flex-col text-primary">
            <Tabs className="flex-grow flex flex-col" defaultValue={"history"}>
              <TabsList className="bg-theme-one grid w-full h-auto grid-cols-3 bottom p-0 m-0 border-0">
                <TabsTrigger className="data-[state=active]:bg-[#E8E3E3] rounded-[inherit]" value="collections">
                  <Icon.Folder size={20} color="#17181D" />
                </TabsTrigger>
                <TabsTrigger className="data-[state=active]:bg-[#E8E3E3] rounded-[inherit]" value="history">
                  <Icon.Clock size={20} color="#17181D" />
                </TabsTrigger>
                <TabsTrigger className="data-[state=active]:bg-[#E8E3E3] rounded-[inherit]" value="tools">
                  <Icon.Aperture size={20} color="#17181D" />
                </TabsTrigger>
              </TabsList>
              {/* <ScrollArea> */}
              <div className='flex-grow'>
                <TabsContent className='mt-0 pt-2' value="collections">
                  <SidebarCollectionSelect
                    scrollClassName='h-[calc(100vh-243px)] overflow-auto scrollbar-hide'
                    userData={userData}
                    collectionGroups={collectionGroups}
                    setCollectionGroups={setCollectionGroups}
                    setCollectionSelected={setCollectionSelected}
                    selectedCollections={selectedCollections}
                  />
                  {/* <div className='flex-grow'/> */}
                </TabsContent>
                <TabsContent value="history" className='mt-0 pt-2'>
                  <SidebarChatHistory
                    scrollClassName='h-[calc(100vh-232px)] scrollbar-hide'
                    // userData={userData}
                    // toolchain_sessions={toolchainSessions}
                    // set_toolchain_sessions={setToolchainSessions}
                    // active_toolchain_session={activeToolchainSession}
                    // set_active_toolchain_session={setActiveToolchainSession}
                  />
                </TabsContent>
                <TabsContent value="tools" className='mt-0 pt-2'>
                  <SidebarToolchains
                    scrollClassName='h-[calc(100vh-232px)] overflow-auto scrollbar-hide'
                    userData={userData}
                    setUserData={setUserData}
                    selected_toolchain={selectedToolchain}
                    set_selected_toolchain={setSelectedToolchain}
                  />
                </TabsContent>
              </div>
              {/* </ScrollArea> */}
            </Tabs>
          </div>
          
          <div className="flex flex-col justify-end w-full p-0 pl-0 pr-5">
            <Button className="w-full justify-start p-0 h-8" variant="link" onClick={() => {
              // TODO: Clear cookies.
            }}>
              <Link href="/auth/login" onClick={() => {
                deleteCookie({ key: "UD"})
                setUserData(undefined);
              }}>
                Logout
              </Link>
            </Button>
            <Button className="w-full justify-start p-0 h-8 text-primary" variant="link">
              <Link className="text-primary" href="/organizations">
                Organizations
              </Link>
            </Button>
            <Button className="w-full justify-start p-0 h-8" variant="link">
              <Link href="/all_pages_panel">
                All Pages
              </Link>
            </Button>
          </div>
        </>
      )}
    </SidebarTemplate>
	);
}
