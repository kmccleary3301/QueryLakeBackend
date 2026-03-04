import { useEffect } from 'react';
import { userDataType, toolchainCategory, toolchain_type, setStateOrCallback } from '@/types/globalTypes';
import { ScrollArea } from '@radix-ui/react-scroll-area';
import { Button } from '@/components/ui/button';
import * as Icon from 'react-feather';
import { cn } from '@/lib/utils';
import { usePathname, useRouter } from 'next/navigation';
import { HoverTextDiv } from '@/components/ui/hover-text-div';
import Link from 'next/link';
import { Copy, Pencil, Trash } from 'lucide-react';
import SidebarEntry from '../manual_components/sidebar-entry-fade';

type SidebarToolchainsProps = {
  userData: userDataType,
  setUserData: React.Dispatch<React.SetStateAction<userDataType | undefined>>,

  selected_toolchain : string | undefined,
  set_selected_toolchain : setStateOrCallback<string | undefined>,

  scrollClassName : string,
}

function ToolchainEntry({
  toolchain,
  selected = false,
  onSelect = () => {},
}:{
  toolchain: toolchain_type,
  selected?: boolean,
  onSelect?: () => void,
}) {
  return (
    <div className="relative not-prose h-10 opacity-100 text-sm rounded-lg hover:bg-accent">
      <div className="group h-full relative rounded-lg flex flex-col justify-center z-5">
        <div className="absolute h-full w-full rounded-[inherit] hover:bg-accent"/>
        <div className="absolute h-full w-full rounded-[inherit] overflow-hidden whitespace-nowrap" onClick={onSelect}>
          <Button variant={"ghost"} className="w-full h-full flex flex-row justify-start p-0 m-0 hover:bg-accent" onClick={onSelect}>
            <div className="w-7 h-full flex flex-col justify-center">
              <div className='w-7 flex flex-row justify-center'>
                {(selected) && (
                  <Icon.Check className='w-3 h-3 text-theme-one'/>
                )}
              </div>
            </div>
            <div className='rounded-[inherit] w-auto flex flex-col justify-center h-full'>
              <div className='flex flex-row'>

                <p className='relative pr-2 overflow-hidden text-nowrap text-sm'>{toolchain.title}</p>
              </div>
            </div>
          </Button>
        </div>
        <div className="absolute h-full w-full bottom-0 right-0 top-0 items-center gap-1.5 rounded-[inherit] overflow-hidden flex flex-row-reverse pointer-events-none">
          {/* <div className='h-full w-full bg-gradient-to-r from-accent/0 from-[calc(100%-80px)] to-card hover:to-accent'/> */}
        </div>
        <div className="absolute h-full rounded-r-[inherit] w-full hidden group-hover:flex group-hover flex-row-reverse overflow-hidden pointer-events-none">
          <div className='h-full flex flex-col justify-center bg-accent z-10'>
            <span className="flex flex-row justify-center pointer-events-auto gap-x-2 pr-2">
              <Link href={`/nodes/node_editor?mode=create&ref=${toolchain.id}`}>
                <button className='h-6 w-4 rounded-full p-0 m-0 text-primary active:text-primary/70'>
                  <Copy className='w-3.5 h-3.5'/>
                </button>
              </Link>
              <Link href={`/nodes/node_editor?mode=edit&t_id=${toolchain.id}`}>
                <button className='h-6 w-4 rounded-full p-0 m-0 text-primary active:text-primary/70'>
                  <Pencil className='w-3.5 h-3.5'/>
                </button>
              </Link>
            </span>
          </div>
          {/* <div className='h-full w-[80px] bg-gradient-to-r from-accent/0 to-card'/> */}
        </div>
      </div>
    </div>
  )
}

export default function SidebarToolchains(props: SidebarToolchainsProps) {
  const pathname = usePathname(),
        router = useRouter();

  // useEffect(() => {
  //   console.log("new userdata selected", props.userData);
  // }, [props.userData]);

  useEffect(() => {
    console.log("Toolchains called");
  }, []);

  const setSelectedToolchain = (toolchain : string) => {
    props.set_selected_toolchain(toolchain);
  };

  return (
    <>
    <div className='pb-0 pt-2'>
      <Link href="/nodes/node_editor?mode=create">
        <Button variant={"ghost"} className="w-full flex flex-row rounded-2xl h-9 items-center justify-center">
          <Icon.Plus className='text-primary pr-[5px]'/>
          <div style={{alignSelf: 'center', justifyContent: 'center'}}>
            <p>{"New Toolchain"}</p>
          </div>
        </Button>
      </Link>
    </div>
    <ScrollArea className={cn("w-full px-[4px] space-y-2", props.scrollClassName)}>
      {props.userData.available_toolchains.map((toolchain_category : toolchainCategory, category_index : number) => (
        <div key={category_index} className='space-y-1'>
          {(toolchain_category.entries.length > 0) && (
            <>
            <p className="w-full text-left text-primary/50 text-sm pt-2">
              {toolchain_category.category}
            </p>
            {toolchain_category.entries.map((toolchain_entry : toolchain_type, index : number) => (
              <SidebarEntry
                key={index}
                className='w-[220px]' 
                title={toolchain_entry.title} 
                displaySelected 
                selected={((props.selected_toolchain !== undefined && props.selected_toolchain !== null) && 
                            props.selected_toolchain === toolchain_entry.id)}
                onSelect={() => {
                  if (toolchain_entry.id !== props.selected_toolchain && pathname?.startsWith("/app/session")) {
                    router.push(`/app/create`);
                  }
                  setSelectedToolchain(toolchain_entry.id);
                }}
              >
                <span className="flex flex-row justify-center pointer-events-auto gap-x-2 pr-2">
                  <Link href={`/nodes/node_editor?mode=create&ref=${toolchain_entry.id}`} className='flex flex-col justify-center'>
                    <button className='h-6 w-4 rounded-full p-0 m-0 text-primary active:text-primary/70'>
                      <Copy className='w-3.5 h-3.5'/>
                    </button>
                  </Link>
                  <Link href={`/nodes/node_editor?mode=edit&t_id=${toolchain_entry.id}`} className='flex flex-col justify-center'>
                    <button className='h-6 w-4 rounded-full p-0 m-0 text-primary active:text-primary/70'>
                      <Pencil className='w-3.5 h-3.5'/>
                    </button>
                  </Link>
                </span>
              </SidebarEntry>
            ))}
            </>
          )}
        </div>
      ))}
    </ScrollArea>
    </>
  );
}