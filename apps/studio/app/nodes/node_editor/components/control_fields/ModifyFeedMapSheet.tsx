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
import { ScrollArea } from "@/components/ui/scroll-area";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import SequenceActionModifier from "./SequenceActionModifier";
import { useState, useEffect } from "react";
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
    output: { color: "bg-red-500", border: "border-red-500", text: "text-red-500", label: "Node Output", icon: "O" },
    input: { color: "bg-blue-500", border: "border-blue-500", text: "text-blue-500", label: "Node Input", icon: "I" },
    state: { color: "bg-yellow-500", border: "border-yellow-500", text: "text-yellow-500", label: "Toolchain State", icon: "S" },
    static: { color: "bg-green-500", border: "border-green-500", text: "text-green-500", label: "Static Value", icon: "V" },
    generic: { color: "bg-purple-500", border: "border-purple-500", text: "text-purple-500", label: "Generic Value", icon: "G" }
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

function SourceConfigurationPanel({ 
  sourceType, 
  onSourceTypeChange,
  outputRoute,
  inputRoute,
  stateRoute,
  staticValue,
  onOutputRouteChange,
  onInputRouteChange,
  onStateRouteChange,
  onStaticValueChange
}: {
  sourceType: FeedMapSourceType;
  onSourceTypeChange: (type: FeedMapSourceType) => void;
  outputRoute: staticRoute;
  inputRoute: staticRoute;
  stateRoute: staticRoute;
  staticValue: any;
  onOutputRouteChange: (route: staticRoute) => void;
  onInputRouteChange: (route: staticRoute) => void;
  onStateRouteChange: (route: staticRoute) => void;
  onStaticValueChange: (value: any) => void;
}) {
  return (
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
        <RadioGroup value={sourceType} onValueChange={onSourceTypeChange} className="space-y-2">
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="output" id="output" />
            <label htmlFor="output" className="text-sm">Node Output</label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="input" id="input" />
            <label htmlFor="input" className="text-sm">Node Input</label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="state" id="state" />
            <label htmlFor="state" className="text-sm">Toolchain State</label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="static" id="static" />
            <label htmlFor="static" className="text-sm">Static Value</label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="generic" id="generic" />
            <label htmlFor="generic" className="text-sm">Generic Value Object</label>
          </div>
        </RadioGroup>

        <Separator />

        {sourceType === "output" && (
          <div className="space-y-2">
            <Label className="text-xs">Output Route</Label>
            <div className="border rounded-md p-2 bg-muted">
              <StaticRouteCreation values={outputRoute} />
            </div>
          </div>
        )}

        {sourceType === "input" && (
          <div className="space-y-2">
            <Label className="text-xs">Input Route</Label>
            <div className="border rounded-md p-2 bg-muted">
              <StaticRouteCreation values={inputRoute} />
            </div>
          </div>
        )}

        {sourceType === "state" && (
          <div className="space-y-2">
            <Label className="text-xs">State Route</Label>
            <div className="border rounded-md p-2 bg-muted">
              <StaticRouteCreation values={stateRoute} />
            </div>
          </div>
        )}

        {sourceType === "static" && (
          <div className="space-y-2">
            <Label className="text-xs">Static Value</Label>
            <Textarea 
              value={typeof staticValue === 'string' ? staticValue : JSON.stringify(staticValue, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  onStaticValueChange(parsed);
                } catch {
                  onStaticValueChange(e.target.value);
                }
              }}
              placeholder="Enter static value (JSON or string)"
              className="text-xs font-mono"
            />
          </div>
        )}

        {sourceType === "generic" && (
          <div className="space-y-2">
            <Label className="text-xs">Generic Value Object</Label>
            <div className="text-xs text-muted-foreground p-2 bg-muted rounded">
              Generic value objects require more complex configuration. This will be implemented in a future update.
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DestinationConfigurationPanel({
  destination,
  route,
  onDestinationChange,
  onRouteChange
}: {
  destination: string;
  route: staticRoute;
  onDestinationChange: (destination: string) => void;
  onRouteChange: (route: staticRoute) => void;
}) {
  return (
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
          <Select value={destination} onValueChange={onDestinationChange}>
            <SelectTrigger className="text-xs">
              <SelectValue placeholder="Select destination" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="<<STATE>>">Toolchain State</SelectItem>
              <SelectItem value="<<USER>>">User Output</SelectItem>
              <SelectItem value="<<FILES>>">File Registry</SelectItem>
              <SelectItem value="custom">Custom Node ID</SelectItem>
            </SelectContent>
          </Select>
          
          {destination === "custom" && (
            <Input 
              placeholder="Enter node ID"
              className="text-xs"
              onChange={(e) => onDestinationChange(e.target.value)}
            />
          )}
        </div>

        <div className="space-y-2">
          <Label className="text-xs">Destination Route (Optional)</Label>
          <div className="border rounded-md p-2 bg-muted">
            <StaticRouteCreation values={route} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function OptionsConfigurationPanel({
  store,
  iterate,
  stream,
  stream_initial_value,
  onStoreChange,
  onIterateChange,
  onStreamChange,
  onStreamInitialValueChange
}: {
  store: boolean;
  iterate: boolean;
  stream: boolean;
  stream_initial_value: any;
  onStoreChange: (store: boolean) => void;
  onIterateChange: (iterate: boolean) => void;
  onStreamChange: (stream: boolean) => void;
  onStreamInitialValueChange: (value: any) => void;
}) {
  return (
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
            checked={store} 
            onCheckedChange={onStoreChange}
          />
          <label htmlFor="store" className="text-xs">
            Store (queue without firing destination)
          </label>
        </div>

        <div className="flex items-center space-x-2">
          <Checkbox 
            id="iterate" 
            checked={iterate} 
            onCheckedChange={onIterateChange}
          />
          <label htmlFor="iterate" className="text-xs">
            Iterate (execute once per item if source is array)
          </label>
        </div>

        <div className="flex items-center space-x-2">
          <Checkbox 
            id="stream" 
            checked={stream} 
            onCheckedChange={onStreamChange}
          />
          <label htmlFor="stream" className="text-xs">
            Stream (handle real-time data updates)
          </label>
        </div>

        {stream && (
          <div className="space-y-2 ml-6">
            <Label className="text-xs">Stream Initial Value</Label>
            <Textarea 
              value={typeof stream_initial_value === 'string' ? stream_initial_value : JSON.stringify(stream_initial_value, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  onStreamInitialValueChange(parsed);
                } catch {
                  onStreamInitialValueChange(e.target.value);
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
  );
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

            {/* Configuration Panels */}
            <SourceConfigurationPanel
              sourceType={editorState.sourceType}
              onSourceTypeChange={(type) => setEditorState(prev => ({ ...prev, sourceType: type }))}
              outputRoute={editorState.outputRoute}
              inputRoute={editorState.inputRoute}
              stateRoute={editorState.stateRoute}
              staticValue={editorState.staticValue}
              onOutputRouteChange={(route) => setEditorState(prev => ({ ...prev, outputRoute: route }))}
              onInputRouteChange={(route) => setEditorState(prev => ({ ...prev, inputRoute: route }))}
              onStateRouteChange={(route) => setEditorState(prev => ({ ...prev, stateRoute: route }))}
              onStaticValueChange={(value) => setEditorState(prev => ({ ...prev, staticValue: value }))}
            />

            <DestinationConfigurationPanel
              destination={editorState.destination}
              route={editorState.route}
              onDestinationChange={(destination) => setEditorState(prev => ({ ...prev, destination }))}
              onRouteChange={(route) => setEditorState(prev => ({ ...prev, route }))}
            />

            <OptionsConfigurationPanel
              store={editorState.store}
              iterate={editorState.iterate}
              stream={editorState.stream}
              stream_initial_value={editorState.stream_initial_value}
              onStoreChange={(store) => setEditorState(prev => ({ ...prev, store }))}
              onIterateChange={(iterate) => setEditorState(prev => ({ ...prev, iterate }))}
              onStreamChange={(stream) => setEditorState(prev => ({ ...prev, stream }))}
              onStreamInitialValueChange={(value) => setEditorState(prev => ({ ...prev, stream_initial_value: value }))}
            />

            {/* Sequence Actions */}
            {editorState.sequence.length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Transformation Sequence</CardTitle>
                  <CardDescription className="text-xs">
                    Actions to transform the data before sending to destination
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {editorState.sequence.map((seq, index) => (
                    <SequenceActionModifier 
                      key={index} 
                      data={seq}
                      deleteSelf={() => {
                        setEditorState(prev => ({
                          ...prev,
                          sequence: prev.sequence.filter((_, i) => i !== index)
                        }));
                      }}
                    />
                  ))}
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full"
                    onClick={() => {
                      // Add new sequence action - for now just add a simple string action
                      setEditorState(prev => ({
                        ...prev,
                        sequence: [...prev.sequence, "newAction"]
                      }));
                    }}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Transformation Step
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Preview */}
            <Accordion type="single" collapsible>
              <AccordionItem value="preview">
                <AccordionTrigger className="text-sm">JSON Preview</AccordionTrigger>
                <AccordionContent>
                  <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                    {JSON.stringify(data, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
} 