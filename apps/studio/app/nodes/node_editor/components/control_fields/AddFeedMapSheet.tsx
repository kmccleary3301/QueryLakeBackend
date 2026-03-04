"use client";
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { feedMapping, staticRoute, valueObj, getNodeOutput, getNodeInput, stateValue, sequenceAction, Condition, conditionBasic } from "@/types/toolchains"
import { StaticRouteCreation } from "./StaticRouteCreation"
import { HoverCard, HoverCardContent } from "@/components/ui/hover-card"
import { HoverCardTrigger } from "@radix-ui/react-hover-card"
import { ContextMenu, ContextMenuContent, ContextMenuTrigger } from "@/components/ui/context-menu"
import Code from "@/components/markdown/code"
import MarkdownCodeBlock from "@/components/markdown/markdown-code-block";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import SequenceActionModifier from "./SequenceActionModifier";
import { useState } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { Plus, Trash2, Info, ChevronRight } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

type FeedMapSourceType = "output" | "input" | "state" | "static" | "generic";

interface FeedMapEditorState {
  sourceType: FeedMapSourceType;
  destination: string;
  route: staticRoute;
  sequence: sequenceAction[];
  stream: boolean;
  stream_initial_value: any;
  store: boolean;
  iterate: boolean;
  condition?: Condition | conditionBasic;
  
  // Source-specific data
  outputRoute: staticRoute;
  inputRoute: staticRoute;
  stateRoute: staticRoute;
  staticValue: any;
  genericValue: valueObj;
}

function SourceTypeIndicator({ type }: { type: FeedMapSourceType }) {
  const configs = {
    output: { color: "bg-red-500", label: "Node Output", icon: "O" },
    input: { color: "bg-blue-500", label: "Node Input", icon: "I" },
    state: { color: "bg-yellow-500", label: "Toolchain State", icon: "S" },
    static: { color: "bg-green-500", label: "Static Value", icon: "V" },
    generic: { color: "bg-purple-500", label: "Generic Value", icon: "G" }
  };
  
  const config = configs[type];
  
  return (
    <div className="flex items-center space-x-2">
      <div className={cn("w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold", config.color)}>
        {config.icon}
      </div>
      <span className="text-sm font-medium">{config.label}</span>
    </div>
  );
}

function DestinationIndicator({ destination }: { destination: string }) {
  const isSpecial = ["<<STATE>>", "<<USER>>", "<<FILES>>"].includes(destination);
  const configs = {
    "<<STATE>>": { color: "bg-yellow-500", icon: "S", label: "Toolchain State" },
    "<<USER>>": { color: "bg-blue-500", icon: "U", label: "User Output" },
    "<<FILES>>": { color: "bg-green-500", icon: "F", label: "File Registry" }
  };
  
  if (isSpecial) {
    const config = configs[destination as keyof typeof configs];
    return (
      <div className="flex items-center space-x-2">
        <div className={cn("w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold", config.color)}>
          {config.icon}
        </div>
        <span className="text-sm font-medium">{config.label}</span>
      </div>
    );
  }
  
  return (
    <div className="flex items-center space-x-2">
      <div className="w-6 h-6 rounded-full bg-gray-500 flex items-center justify-center text-white text-xs font-bold">
        N
      </div>
      <span className="text-sm font-medium">Node: {destination}</span>
    </div>
  );
}

export default function AddFeedMapSheet({
  data,
  className,
  children,
}: {
  data?: feedMapping,
  className?: string,
  children?: React.ReactNode
}) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className={className}>
          {children}
        </div>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        {(data && data.route !== undefined && data.route !== null) && (
          <StaticRouteCreation values={data.route} className=""/>
        )}
      </PopoverContent>
    </Popover>
  )
}

export function ModifyFeedMapSheet({
  data,
  className,
  children,
  onSave,
  ...props
}: {
  data?: feedMapping,
  className?: string,
  children?: React.ReactNode,
  onSave?: (feedMap: feedMapping) => void
}) {
  const [editorState, setEditorState] = useState<FeedMapEditorState>(() => {
    if (!data) {
      return {
        sourceType: "output" as FeedMapSourceType,
        destination: "<<STATE>>",
        route: [],
        sequence: [],
        stream: false,
        stream_initial_value: "",
        store: false,
        iterate: false,
        outputRoute: [],
        inputRoute: [],
        stateRoute: [],
        staticValue: "",
        genericValue: { type: "staticValue", value: "" }
      };
    }

    // Determine source type from existing data
    let sourceType: FeedMapSourceType = "output";
    let outputRoute: staticRoute = [];
    let inputRoute: staticRoute = [];
    let stateRoute: staticRoute = [];
    let staticValue: any = "";

    if ("getFromOutputs" in data) {
      sourceType = "output";
      outputRoute = data.getFromOutputs.route;
    } else if ("getFromInputs" in data) {
      sourceType = "input";
      inputRoute = data.getFromInputs.route;
    } else if ("getFromState" in data) {
      sourceType = "state";
      stateRoute = data.getFromState.route;
    } else if ("value" in data) {
      sourceType = "static";
      staticValue = data.value;
    } else if ("getFrom" in data) {
      sourceType = "generic";
    }

    return {
      sourceType,
      destination: data.destination,
      route: data.route || [],
      sequence: data.sequence || [],
      stream: data.stream || false,
      stream_initial_value: data.stream_initial_value || "",
      store: data.store || false,
      iterate: data.iterate || false,
      outputRoute,
      inputRoute,
      stateRoute,
      staticValue,
      genericValue: { type: "staticValue", value: "" }
    };
  });

  const handleSave = () => {
    // Convert editor state back to feedMapping
    const baseFeedMap = {
      destination: editorState.destination,
      route: editorState.route.length > 0 ? editorState.route : undefined,
      sequence: editorState.sequence.length > 0 ? editorState.sequence : undefined,
      stream: editorState.stream || undefined,
      stream_initial_value: editorState.stream ? editorState.stream_initial_value : undefined,
      store: editorState.store || undefined,
      iterate: editorState.iterate || undefined,
      condition: editorState.condition
    };

    let feedMap: feedMapping;
    
    switch (editorState.sourceType) {
      case "output":
        feedMap = {
          ...baseFeedMap,
          getFromOutputs: { route: editorState.outputRoute }
        } as any;
        break;
      case "input":
        feedMap = {
          ...baseFeedMap,
          getFromInputs: { route: editorState.inputRoute }
        } as any;
        break;
      case "state":
        feedMap = {
          ...baseFeedMap,
          getFromState: { route: editorState.stateRoute }
        } as any;
        break;
      case "static":
        feedMap = {
          ...baseFeedMap,
          value: editorState.staticValue
        } as any;
        break;
      case "generic":
        feedMap = {
          ...baseFeedMap,
          getFrom: editorState.genericValue
        } as any;
        break;
    }

    onSave?.(feedMap);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className={className}>
          {children}
        </div>
      </PopoverTrigger>
      <PopoverContent className="w-auto pr-0 py-0" side="left" align="start">
        <ScrollArea className="h-[600px]">
          <div className="w-[500px] pr-5 py-4 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-semibold">Feed Mapping Editor</h3>
                <div className="flex items-center space-x-4">
                  <SourceTypeIndicator type={editorState.sourceType} />
                  <span className="text-muted-foreground">→</span>
                  <DestinationIndicator destination={editorState.destination} />
                </div>
              </div>
              <Button size="sm" onClick={handleSave}>
                Save Changes
              </Button>
            </div>

            <Separator />

            {/* Source Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <Info className="w-4 h-4 mr-2" />
                  Data Source
                </CardTitle>
                <CardDescription className="text-xs">
                  Choose where this feed mapping gets its data from
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <RadioGroup value={editorState.sourceType} onValueChange={(type: FeedMapSourceType) => setEditorState(prev => ({ ...prev, sourceType: type }))} className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="output" id="output" />
                    <Label htmlFor="output" className="text-sm">Node Output</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="input" id="input" />
                    <Label htmlFor="input" className="text-sm">Node Input</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="state" id="state" />
                    <Label htmlFor="state" className="text-sm">Toolchain State</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="static" id="static" />
                    <Label htmlFor="static" className="text-sm">Static Value</Label>
                  </div>
                </RadioGroup>

                <Separator />

                {editorState.sourceType === "static" && (
                  <div className="space-y-2">
                    <Label className="text-xs">Static Value</Label>
                    <Textarea 
                      value={typeof editorState.staticValue === 'string' ? editorState.staticValue : JSON.stringify(editorState.staticValue, null, 2)}
                      onChange={(e) => {
                        try {
                          const parsed = JSON.parse(e.target.value);
                          setEditorState(prev => ({ ...prev, staticValue: parsed }));
                        } catch {
                          setEditorState(prev => ({ ...prev, staticValue: e.target.value }));
                        }
                      }}
                      placeholder="Enter static value (JSON or string)"
                      className="text-xs font-mono"
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Destination Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <ChevronRight className="w-4 h-4 mr-2" />
                  Destination
                </CardTitle>
                <CardDescription className="text-xs">
                  Where should this data be sent?
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Destination Target</Label>
                  <Select value={editorState.destination} onValueChange={(destination) => setEditorState(prev => ({ ...prev, destination }))}>
                    <SelectTrigger className="text-xs">
                      <SelectValue placeholder="Select destination" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="<<STATE>>">Toolchain State</SelectItem>
                      <SelectItem value="<<USER>>">User Output</SelectItem>
                      <SelectItem value="<<FILES>>">File Registry</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Options Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Execution Options</CardTitle>
                <CardDescription className="text-xs">
                  Control how this feed mapping behaves
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="store" 
                    checked={editorState.store} 
                    onCheckedChange={(store) => setEditorState(prev => ({ ...prev, store: !!store }))}
                  />
                  <Label htmlFor="store" className="text-xs">
                    Store (queue without firing destination)
                  </Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="iterate" 
                    checked={editorState.iterate} 
                    onCheckedChange={(iterate) => setEditorState(prev => ({ ...prev, iterate: !!iterate }))}
                  />
                  <Label htmlFor="iterate" className="text-xs">
                    Iterate (execute once per item if source is array)
                  </Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="stream" 
                    checked={editorState.stream} 
                    onCheckedChange={(stream) => setEditorState(prev => ({ ...prev, stream: !!stream }))}
                  />
                  <Label htmlFor="stream" className="text-xs">
                    Stream (handle real-time data updates)
                  </Label>
                </div>

                {editorState.stream && (
                  <div className="space-y-2 ml-6">
                    <Label className="text-xs">Stream Initial Value</Label>
                    <Textarea 
                      value={typeof editorState.stream_initial_value === 'string' ? editorState.stream_initial_value : JSON.stringify(editorState.stream_initial_value, null, 2)}
                      onChange={(e) => {
                        try {
                          const parsed = JSON.parse(e.target.value);
                          setEditorState(prev => ({ ...prev, stream_initial_value: parsed }));
                        } catch {
                          setEditorState(prev => ({ ...prev, stream_initial_value: e.target.value }));
                        }
                      }}
                      placeholder="Initial value for streaming (usually empty string or array)"
                      className="text-xs font-mono"
                      rows={2}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Legacy Raw Value Display */}
            <Accordion type="single" collapsible>
              <AccordionItem value="preview">
                <AccordionTrigger className="text-sm">Raw JSON Preview</AccordionTrigger>
                <AccordionContent>
                  <MarkdownCodeBlock text={JSON.stringify(data, null, 4)} lang="JSON" finished/>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </ScrollArea>
      </PopoverContent>
    </Popover>
  )
}

