"use client";
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
	Sheet,
	SheetClose,
	SheetContent,
	SheetDescription,
	SheetFooter,
	SheetHeader,
	SheetTitle,
	SheetTrigger,
} from "@/components/ui/sheet"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
	DropdownMenuItem
} from "@/components/ui/dropdown-menu"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { INPUT_COMPONENT_FIELDS, configEntry, configEntryFieldType, inputComponentConfig, inputComponents, inputEvent, inputMapping } from "@/types/toolchain-interface"
import { ChevronDown, ChevronUp, Plus, Trash2, Info, ChevronRight } from "lucide-react"
import { useEffect, useState, Fragment, useCallback, ChangeEvent } from "react"
import CompactInput from "@/components/ui/compact-input"
import { HoverTextDiv } from "@/components/ui/hover-text-div";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Toggle } from "@/components/ui/toggle";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import MarkdownCodeBlock from "@/components/markdown/markdown-code-block";

export default function InputComponentSheet({
	value,
	type,
	onChange,
	onDelete,
	children
}:{
	value: inputMapping,
	type: inputComponents,
	onChange: (config : inputMapping) => void,
	onDelete: () => void,
	children: React.ReactNode
}) {
	
	const [componentConfigFields, setComponentConfigFields] = useState<inputComponentConfig>(INPUT_COMPONENT_FIELDS[type]);

  const updateConfigFields = (input_mapping_value : inputMapping) => {
    let newConfigFields : {[key: string]: configEntry} = {};
    (INPUT_COMPONENT_FIELDS[type].config || []).forEach((c : configEntryFieldType) => {
      newConfigFields[c.name] = {name : c.name, value: c.default};
    });
    input_mapping_value.config.forEach((c : configEntry) => {
      newConfigFields[c.name] = {name : c.name, value: c.value};
    });

    return {
      ...input_mapping_value,
      config: (INPUT_COMPONENT_FIELDS[type].config || []).map((c : configEntryFieldType) => (newConfigFields[c.name]))
    };
  }

  const [actingValue, setActingValue] = useState<inputMapping>(updateConfigFields(JSON.parse(JSON.stringify(value)) as inputMapping));

	const resetActingValue = useCallback(() => {setActingValue(JSON.parse(JSON.stringify(value)))}, [value]);
	const setHook = useCallback((hook : inputEvent, index : number) => {
		setActingValue({
			...actingValue, 
			hooks: [
				...actingValue.hooks.slice(0, index),
				hook,
				...actingValue.hooks.slice(index+1)
			]
		});
	}, [actingValue]);

  const setConfigValue = (p : boolean | string | number, index : number) => {
    const field_name = (INPUT_COMPONENT_FIELDS[type].config || [])[index].name;
    if (!field_name) return;

    setActingValue((oldValue : inputMapping) => {return {
      ...oldValue, 
      config: [
        ...oldValue.config.slice(0, index),
        {name: field_name, value: p},
        ...oldValue.config.slice(index+1)
      ]
    }});
  };

	useEffect(() => {setComponentConfigFields(INPUT_COMPONENT_FIELDS[type])}, [type]);
	useEffect(() => {resetActingValue()}, [value, resetActingValue]);
  
	return (
		<Sheet>
			<SheetTrigger asChild>
				{children}
			</SheetTrigger>
			<SheetContent className="h-full flex flex-col p-0 right-0 w-full" onContextMenu={(e) => {e.preventDefault()}}>
				<SheetHeader className="pt-6 px-6">
					<SheetTitle>{`Configure \`${type}\` Component`}</SheetTitle>
					<SheetDescription>
						Configure input component for toolchain interface.
					</SheetDescription>
				</SheetHeader>

        <ScrollArea className="flex-grow pl-6 pr-2 py-2">
          <div className="space-y-4 py-4 w-[90%]">
            {/* Styling Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <Info className="w-4 h-4 mr-2" />
                  Component Styling
                </CardTitle>
                <CardDescription className="text-xs">
                  Configure the visual appearance of this component
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Tailwind Classes</Label>
                  <CompactInput 
                    onChange={(e : ChangeEvent<HTMLInputElement>) => setActingValue({...actingValue, tailwind: e.target.value})}
                    placeholder='Component Tailwind' 
                    className='h-9 w-full'
                    defaultValue={actingValue.tailwind}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Hooks Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <ChevronRight className="w-4 h-4 mr-2" />
                  Event Hooks
                </CardTitle>
                <CardDescription className="text-xs">
                  Configure when and how this component triggers events
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Available Hooks</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger className="flex items-center gap-1">
                      <Plus className="w-4 h-4 text-primary"/>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      {componentConfigFields.hooks.map((hook, index) => (
                        <DropdownMenuItem key={index} onClick={() => {
                          setHook({
                            hook: hook, 
                            target_event: "", 
                            fire_index: Math.max(...actingValue.hooks.map((e) => e.fire_index), 0) + 1,
                            target_route: "", 
                            store: false
                          }, 
                          actingValue.hooks.length);
                        }}>{hook}</DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                <div className="space-y-4">
                  {actingValue.hooks.map((hookLocal, index) => (
                    <div key={index} className="space-y-2 p-2 border rounded-lg">
                      <div className="flex items-center justify-between">
                        <div className="flex flex-wrap items-center space-x-2">
                          <span className="text-sm font-medium">{hookLocal.hook}</span>
                          <HoverTextDiv hint="Order of hook firing. Can be used to fire inputs together.">
                            <div className="flex items-center space-x-1">
                              <Button className="p-0 h-6 w-6" variant="ghost" onClick={() => {
                                const newIndex = (hookLocal.fire_index) % (actingValue.hooks.length);
                                setHook({...hookLocal, fire_index: newIndex + 1}, index);
                              }}>
                                <ChevronUp className="w-4 h-4 text-primary"/>
                              </Button>
                              <span className="text-xs font-bold">{hookLocal.fire_index}</span>
                              <Button className="p-0 h-6 w-6" variant="ghost" onClick={() => {
                                const newIndex = (hookLocal.fire_index - 2 + actingValue.hooks.length) % (actingValue.hooks.length); 
                                setHook({...hookLocal, fire_index: newIndex + 1}, index);
                              }}>
                                <ChevronDown className="w-4 h-4 text-primary"/>
                              </Button>
                            </div>
                          </HoverTextDiv>
                        </div>
                        <div className="flex items-center space-x-2">
                          <ToggleGroup
                            type="single"
                            onValueChange={(value : "" | "store") => {
                              setHook({...hookLocal, store: (value === "store")}, index);
                            }}
                            className='flex flex-row justify-between'
                            value={(hookLocal.store) ? "store" : ""}
                          >
                            <ToggleGroupItem value="store" aria-label="Store" variant={"outline"} className="text-xs">
                              Store
                            </ToggleGroupItem>
                          </ToggleGroup>
                          <Button size="icon" variant="ghost" onClick={() => {
                            setActingValue({
                              ...actingValue, 
                              hooks: [
                                ...actingValue.hooks.slice(0, index),
                                ...actingValue.hooks.slice(index+1)
                              ]
                            });
                          }}>
                            <Trash2 className="w-4 h-4 text-primary" />
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          value={hookLocal.target_event} 
                          onChange={(event) => {
                            setHook({...hookLocal, target_event: event.target.value}, index);
                          }}
                          placeholder="Target Event"
                          spellCheck={false}
                          className="text-xs"
                        />
                        <Input
                          value={hookLocal.target_route} 
                          onChange={(event) => {
                            setHook({...hookLocal, target_route: event.target.value}, index);
                          }}
                          placeholder="Target Route"
                          spellCheck={false}
                          className="text-xs"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Component Configuration */}
            {componentConfigFields.config && componentConfigFields.config.length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Component Configuration</CardTitle>
                  <CardDescription className="text-xs">
                    Configure component-specific settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    {componentConfigFields.config.map((configEntry : configEntryFieldType, index : number) => (
                      <div key={index} className="space-y-2">
                        <Label className="text-xs">{configEntry.name}</Label>
                        {configEntry.type === "boolean" && (
                          <Toggle className="w-full" pressed={actingValue.config[index]?.value as boolean} onPressedChange={(p : boolean) => {
                            setConfigValue(p, index);
                          }}>
                            {configEntry.name}
                          </Toggle>
                        )}
                        {configEntry.type === "string" && (
                          <CompactInput 
                            className="w-full" 
                            value={actingValue.config[index]?.value as string} 
                            placeholder={configEntry.name} 
                            onChange={(e: ChangeEvent<HTMLInputElement>) => {
                              setConfigValue(e.target.value, index);
                            }}
                          />
                        )}
                        {configEntry.type === "long_string" && (
                          <Textarea 
                            className="w-full resize-none" 
                            spellCheck={false} 
                            value={actingValue.config[index]?.value as string} 
                            placeholder={configEntry.name}
                            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => {
                              setConfigValue(e.target.value, index);
                            }}
                          />
                        )}
                        {configEntry.type === "number" && (
                          <CompactInput 
                            className="w-full" 
                            value={actingValue.config[index]?.value as number} 
                            placeholder={configEntry.name} 
                            type="number" 
                            onChange={(e: ChangeEvent<HTMLInputElement>) => {
                              setConfigValue(parseFloat(e.target.value), index);
                            }}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Raw JSON Preview */}
            <Accordion type="single" collapsible>
              <AccordionItem value="preview">
                <AccordionTrigger className="text-sm">Raw JSON Preview</AccordionTrigger>
                <AccordionContent>
                  <MarkdownCodeBlock text={JSON.stringify(actingValue, null, 4)} lang="JSON" finished/>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </ScrollArea>

				<SheetFooter>
					<SheetClose asChild>
						<Button type="submit" className="w-full" onClick={() => onChange(actingValue)}>Save Changes</Button>
					</SheetClose>
				</SheetFooter>
			</SheetContent>
		</Sheet>
	)
}
