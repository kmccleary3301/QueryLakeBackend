"use client";

import { useContextAction } from '@/app/context-provider';
import {
	ContextMenu,
	ContextMenuCheckboxItem,
	ContextMenuContent,
	ContextMenuItem,
	ContextMenuLabel,
	ContextMenuRadioGroup,
	ContextMenuRadioItem,
	ContextMenuSeparator,
	ContextMenuShortcut,
	ContextMenuSub,
	ContextMenuSubContent,
	ContextMenuSubTrigger,
	ContextMenuTrigger,
} from '@/components/ui/context-menu';
import {
  HoverCard,
  HoverCardContent,  
  HoverCardTrigger,
} from '@/components/ui/hover-card';
// import { ScrollArea } from '@radix-ui/react-scroll-area';
import { ScrollArea } from '@/components/ui/scroll-area';
import { APIFunctionSpec } from '@/types/globalTypes';
import { toolchainNode } from '@/types/toolchains';
import { MouseEvent, MouseEventHandler, useCallback, useState } from 'react';
import { Node, ReactFlowInstance } from 'reactflow';


export default function ContextMenuWrapper({ 
  reactFlowInstance,
  setNodes,
  getId,
	children
}: {
  reactFlowInstance: ReactFlowInstance<any, any> | null;
  setNodes: React.Dispatch<React.SetStateAction<Node<object, string | undefined>[]>>;
  getId: () => string;
	children: React.ReactNode;
}) {

  const { apiFunctionSpecs } = useContextAction();
  const [hoveredFunction, setHoveredFunction] = useState<APIFunctionSpec | null>(null);

  const onAddTestItem = useCallback(
    (event : MouseEvent<HTMLDivElement, globalThis.MouseEvent>) => {
      // event.preventDefault();
      if (!reactFlowInstance) {
        console.log("No reactFlowInstance; exiting");
        return;
      }
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: "dndnode_test_"+getId(),
        position: position,
        data: { icon: <div/>, title: 'fullBundle' },
        type: 'turbo',
      };

      setNodes((nds) => [...nds, newNode]);
    },
    [reactFlowInstance, getId, setNodes],
  );

  const onAddAPIFunctionNode = useCallback(
    (event : MouseEvent<HTMLDivElement, globalThis.MouseEvent>, api_func: APIFunctionSpec) => {
      // event.preventDefault();
      if (!reactFlowInstance) {
        console.log("No reactFlowInstance; exiting");
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });


      const new_node_data : toolchainNode = {
        id: api_func.api_function_id + "_" + getId(),
        api_function: api_func.api_function_id,
        feed_mappings: [],
        input_arguments: api_func.function_args.map((arg) => ({
          key: arg.keyword,
          ...(arg.default_value?{ default_value: arg.default_value }:{}),
          ...(arg.type_hint?{ type_hint: arg.type_hint }:{}),
        }))
      }

      const newNode = {
        id: getId(),
        position: position,
        data: new_node_data,
        type: 'toolchain',
      };

      setNodes((nds) => [...nds, newNode]);
    },
    [reactFlowInstance, getId, setNodes],
  );


  return (
		<ContextMenu modal={true}>
			<ContextMenuTrigger className="flex z-5 h-full w-full items-center justify-center rounded-md border border-dashed text-sm">
				{children}
			</ContextMenuTrigger>
			<ContextMenuContent className="w-64">
				<ContextMenuItem inset onClick={(event) => {
          onAddTestItem(event);
        }}>
          Add Test Item
				</ContextMenuItem>
				{/* <ContextMenuItem inset disabled>
					Forward
					<ContextMenuShortcut>⌘]</ContextMenuShortcut>
				</ContextMenuItem> */}
				<ContextMenuItem inset>
					Add Empty Node
				</ContextMenuItem>
				<ContextMenuSub>
					<ContextMenuSubTrigger inset>Add API Call</ContextMenuSubTrigger>
					<ContextMenuSubContent className="">
            <HoverCard>
              {hoveredFunction && (
                <HoverCardContent
                  side="left"
                  align="start"
                  forceMount
                  className="w-[350px] max-w-[350px] z-50"
                >
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-sm font-semibold break-words">{hoveredFunction.api_function_id}</h4>
                      <p className="text-xs text-muted-foreground break-words">{hoveredFunction.endpoint}</p>
                    </div>
                    {hoveredFunction.description && (
                      <div>
                        <h5 className="text-xs font-medium text-muted-foreground mb-1">Description</h5>
                        <p className="text-xs break-words">{hoveredFunction.description}</p>
                      </div>
                    )}
                    {hoveredFunction.function_args && hoveredFunction.function_args.length > 0 && (
                      <div>
                        <h5 className="text-xs font-medium text-muted-foreground mb-2">Arguments</h5>
                        <div className="space-y-1.5">
                          {hoveredFunction.function_args.map((arg, index) => (
                            <div key={index} className="flex flex-col gap-1">
                              <div className="flex items-start gap-2 min-w-0">
                                <code className="text-xs bg-muted px-1 py-0.5 rounded flex-shrink-0">{arg.keyword}</code>
                                {arg.type_hint && (
                                  <span className="text-xs text-muted-foreground break-all overflow-hidden text-ellipsis leading-relaxed min-w-0 flex-1">
                                    {arg.type_hint}
                                  </span>
                                )}
                              </div>
                              {arg.default_value && (
                                <div className="text-xs text-muted-foreground ml-1 break-words">
                                  default: {JSON.stringify(arg.default_value)}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </HoverCardContent>
              )}
              <div className="relative">
                <HoverCardTrigger asChild>
                  <div className="absolute inset-0 pointer-events-none" />
                </HoverCardTrigger>
                <ScrollArea className='h-[400px]'>
                  {apiFunctionSpecs?.sort((a, b) => a.api_function_id.localeCompare(b.api_function_id)).map((spec, index) => (
                    <ContextMenuItem 
                      inset 
                      key={index} 
                      className='pl-2 mr-2.5' 
                      onClick={(event) => {
                        onAddAPIFunctionNode(event, spec);
                      }}
                      onMouseEnter={() => setHoveredFunction(spec)}
                      onMouseLeave={() => setHoveredFunction(null)}
                    >
                      {spec.api_function_id}
                    </ContextMenuItem>
                  ))}
                </ScrollArea>
              </div>
            </HoverCard>
					</ContextMenuSubContent>
				</ContextMenuSub>
			</ContextMenuContent>
		</ContextMenu>
  );
}
