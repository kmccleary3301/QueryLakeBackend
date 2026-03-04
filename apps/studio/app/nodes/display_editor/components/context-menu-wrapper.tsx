"use client";
import { ViewVerticalIcon, ViewHorizontalIcon } from '@radix-ui/react-icons'

import {
	ContextMenu,
	ContextMenuContent,
	ContextMenuSeparator,
	ContextMenuItem,
	ContextMenuLabel,
	ContextMenuShortcut,
	ContextMenuSub,
	ContextMenuSubContent,
	ContextMenuSubTrigger,
	ContextMenuTrigger,
} from '@/components/ui/context-menu';
import { 
	AlignLeft,
	AlignRight,
	AlignCenter,
	AlignJustify,
	Wand
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { 
	alignType, 
	displayComponents, 
	displayMapping,
	DISPLAY_COMPONENTS,
	INPUT_COMPONENTS,
	contentMapping,
  contentDiv,
  INPUT_COMPONENT_FIELDS,
  configEntryFieldType
} from '@/types/toolchain-interface';
import CompactInput from '@/components/ui/compact-input';
import { ChangeEvent, useRef } from 'react';
import { config } from 'process';
import { cn } from '@/lib/utils';


export function ContextMenuViewportWrapper({
	onSplit,
	onCollapse,
	onAlign,
	setTailwind,
	addComponent,
	align,
	tailwind,
	headerAvailable = true,
	footerAvailable = true,
  className = "",
	children
}: {
	onSplit : (split_type : "horizontal" | "vertical" | "header" | "footer", count: number) => void,
	onCollapse : () => void,
	onAlign: (a : alignType) => void,
	setTailwind: (t : string) => void,
	addComponent: (component : contentMapping | contentDiv) => void,
	align: alignType,
	tailwind: string,
	headerAvailable?: boolean,
	footerAvailable?: boolean,
  className?: string,
	children: React.ReactNode,
}) {
	const tailwindRef= useRef("");

  return (
		<ContextMenu>
			<ContextMenuTrigger className={cn("z-5 items-center justify-center text-sm", className)}>
				{children}
			</ContextMenuTrigger>
			<ContextMenuContent className="space-y-1 pt-6 p-2">
				<ToggleGroup 
					type="single" 
					onValueChange={(value : alignType | "") => {
						if (value !== "") onAlign(value);
					}}
					className='flex flex-row justify-between'
					value={align}
				>
					<ToggleGroupItem value="left" aria-label="Align left">
						<AlignLeft className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
					<ToggleGroupItem value="center" aria-label="Align center">
						<AlignCenter className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
					<ToggleGroupItem value="right" aria-label="Align right">
						<AlignRight className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
				</ToggleGroup>
				<div className='pt-2 pb-2 flex flex-row space-x-1'>
					<CompactInput 
						onChange={(e : ChangeEvent<HTMLInputElement>) => tailwindRef.current = e.target.value}
						placeholder='Inner Tailwind' 
						className='h-9 w-40'
						defaultValue={tailwind}
					/>
					<Button className='h-9 px-0 w-9' variant={"ghost"} onClick={() => setTailwind(tailwindRef.current)}>
						<Wand className='h-4 w-4 text-primary'/>
					</Button>
				</div>
				<ContextMenuSeparator/>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
					<p className='text-primary/0 text-xs'>.</p>Add Display
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
            <ContextMenuItem inset onClick={() => (addComponent({
              type: "div",
              align: "center",
              tailwind: "min-w-[20px] min-h-[20px]", 
              mappings: []
            }))}>
              Div
            </ContextMenuItem>
						{DISPLAY_COMPONENTS.map((component, index) => (
							<ContextMenuItem inset key={index} onClick={() => (addComponent({
								display_route: [],
								display_as: component
							}))}>
								{component}
							</ContextMenuItem>
						))}
          </ContextMenuSubContent>
        </ContextMenuSub>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
					<p className='text-primary/0 text-xs'>.</p>Add Input
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
						{INPUT_COMPONENTS.map((component, index) => (
							<ContextMenuItem inset key={index} onClick={() => (addComponent({
								display_as: component,
								hooks: [],
								config: (INPUT_COMPONENT_FIELDS[component].config || []).map((c : configEntryFieldType) => ({
                  name: c.name,
                  value: c.default
                })),
								tailwind: ""
							}))}>
								{component}
							</ContextMenuItem>
						))}
          </ContextMenuSubContent>
        </ContextMenuSub>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
					<p className='text-primary/0 text-xs'>.</p>Split Horizontal <div className='w-2'/><ViewVerticalIcon viewBox="0 0 15.5 15"/>
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
            <ContextMenuItem inset onClick={() => (onSplit("horizontal", 2))}>2</ContextMenuItem>
            <ContextMenuItem inset onClick={() => (onSplit("horizontal", 3))}>3</ContextMenuItem>
            <ContextMenuItem inset onClick={() => (onSplit("horizontal", 4))}>4</ContextMenuItem>
						<ContextMenuItem inset onClick={() => (onSplit("horizontal", 5))}>5</ContextMenuItem>
          </ContextMenuSubContent>
        </ContextMenuSub>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
						<p className='text-primary/0 text-xs'>.</p>Split Vertical <div className='w-2'/><ViewHorizontalIcon/>
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
            <ContextMenuItem inset onClick={() => (onSplit("vertical", 2))}>2</ContextMenuItem>
            <ContextMenuItem inset onClick={() => (onSplit("vertical", 3))}>3</ContextMenuItem>
            <ContextMenuItem inset onClick={() => (onSplit("vertical", 4))}>4</ContextMenuItem>
						<ContextMenuItem inset onClick={() => (onSplit("vertical", 5))}>5</ContextMenuItem>
          </ContextMenuSubContent>
        </ContextMenuSub>
				{/* <ContextMenuItem inset className='pr-2' onClick={() => (onSplit("horizontal"))}>
					Split Horizontal <div className='w-2'/><ViewVerticalIcon viewBox="0 0 15.5 15"/>
				</ContextMenuItem> */}
				{/* <ContextMenuItem inset onClick={() => (onSplit("vertical"))}>
					Split Vertical <div className='w-2'/><ViewHorizontalIcon/>
				</ContextMenuItem> */}
				{headerAvailable && (<ContextMenuItem inset onClick={() => (onSplit("header", 0))}>
					<p className='text-primary/0 text-xs'>.</p>Add Header
				</ContextMenuItem>)}
				{footerAvailable && (<ContextMenuItem inset onClick={() => (onSplit("footer", 0))}>
					<p className='text-primary/0 text-xs'>.</p>Add Footer
				</ContextMenuItem>)}
				<ContextMenuItem inset onClick={onCollapse}>
					<p className='text-primary/0 text-xs'>.</p>Delete
				</ContextMenuItem>
			</ContextMenuContent>
		</ContextMenu>
  );
}

export function ContextMenuHeaderWrapper({
	onCollapse,
	onAlign,
	setTailwind,
	addComponent,
	align,
	tailwind,
  className = "",
	children
}: {
	onCollapse : () => void,
	onAlign: (a : alignType) => void,
	setTailwind: (t : string) => void,
	addComponent: (component : (contentMapping | contentDiv)) => void,
	align: alignType,
	tailwind: string,
  className?: string,
	children: React.ReactNode,
}) {

	const tailwindRef= useRef("");

  return (
		<ContextMenu>
			<ContextMenuTrigger className={cn("flex z-5 items-center justify-center text-sm", className)}>
				{children}
			</ContextMenuTrigger>
			<ContextMenuContent className="space-y-2 pt-4 p-2">
				<ToggleGroup 
					type="single" 
					onValueChange={(value : alignType | "") => {
						if (value !== "") onAlign(value);
					}}
					className='flex flex-row justify-between'
					value={align}
				>
					<ToggleGroupItem value="left" aria-label="Align left">
						<AlignLeft className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
					<ToggleGroupItem value="center" aria-label="Align center">
						<AlignCenter className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
					<ToggleGroupItem value="justify" aria-label="Align justify">
						<AlignJustify className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
					<ToggleGroupItem value="right" aria-label="Align right">
						<AlignRight className="h-4 w-4 text-primary" />
					</ToggleGroupItem>
				</ToggleGroup>
				<div className='pt-2 pb-2 flex flex-row space-x-1'>
					<CompactInput 
						onChange={(e : ChangeEvent<HTMLInputElement>) => tailwindRef.current = e.target.value}
						placeholder='Inner Tailwind' 
						className='h-9 w-40'
						defaultValue={tailwind}
					/>
					<Button className='h-9 px-0 w-9' variant={"ghost"} onClick={() => setTailwind(tailwindRef.current)}>
						<Wand className='h-4 w-4 text-primary'/>
					</Button>
				</div>
				<ContextMenuSeparator/>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
					<p className='text-primary/0 text-xs'>.</p>Add Display
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
            <ContextMenuItem inset onClick={() => (addComponent({
              type: "div",
              align: "center",
              tailwind: "min-w-[20px] min-h-[20px]", 
              mappings: []
            }))}>
              Div
            </ContextMenuItem>
						{DISPLAY_COMPONENTS.map((component, index) => (
							<ContextMenuItem inset key={index} onClick={() => (addComponent({
								display_route: [],
								display_as: component
							}))}>
								{component}
							</ContextMenuItem>
						))}
          </ContextMenuSubContent>
        </ContextMenuSub>
				<ContextMenuSub>
          <ContextMenuSubTrigger inset>
					<p className='text-primary/0 text-xs'>.</p>Add Input
					</ContextMenuSubTrigger>
          <ContextMenuSubContent className="w-48">
						{INPUT_COMPONENTS.map((component, index) => (
							<ContextMenuItem inset key={index} onClick={() => (addComponent({
								display_as: component,
								hooks: [],
								config: [],
								tailwind: ""
							}))}>
								{component}
							</ContextMenuItem>
						))}
          </ContextMenuSubContent>
        </ContextMenuSub>
				<ContextMenuItem inset onClick={onCollapse}>
					Delete
				</ContextMenuItem>
			</ContextMenuContent>
		</ContextMenu>
  );
}