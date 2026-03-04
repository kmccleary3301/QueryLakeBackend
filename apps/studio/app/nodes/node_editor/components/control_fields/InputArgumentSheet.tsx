"use client";
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { nodeInputArgument, stateValue, getFiles } from "@/types/toolchains"
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { Info, Settings, User, Server, Database, FileText } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

type InputSourceType = "static" | "user" | "server" | "state" | "files";

interface InputArgumentEditorState {
  key: string;
  sourceType: InputSourceType;
  staticValue: any;
  from_user: boolean;
  from_server: boolean;
  from_state?: stateValue;
  from_files?: getFiles;
  optional: boolean;
  type_hint: string;
}

function SourceTypeIndicator({ type }: { type: InputSourceType }) {
  const configs = {
    static: { color: "bg-green-500", label: "Static Value", icon: Settings },
    user: { color: "bg-blue-500", label: "User Input", icon: User },
    server: { color: "bg-purple-500", label: "Server Args", icon: Server },
    state: { color: "bg-yellow-500", label: "Toolchain State", icon: Database },
    files: { color: "bg-orange-500", label: "File Input", icon: FileText }
  };
  
  const config = configs[type];
  const IconComponent = config.icon;
  
  return (
    <div className="flex items-center space-x-2">
      <div className={cn("w-6 h-6 rounded-full flex items-center justify-center text-white", config.color)}>
        <IconComponent className="w-3 h-3" />
      </div>
      <span className="text-sm font-medium">{config.label}</span>
    </div>
  );
}

export function InputArgumentSheet({
  data,
  className,
  children,
  onSave,
  ...props
}: {
  data?: nodeInputArgument,
  className?: string,
  children?: React.ReactNode,
  onSave?: (inputArg: nodeInputArgument) => void
}) {
  const [editorState, setEditorState] = useState<InputArgumentEditorState>(() => {
    if (!data) {
      return {
        key: "",
        sourceType: "static" as InputSourceType,
        staticValue: "",
        from_user: false,
        from_server: false,
        optional: false,
        type_hint: ""
      };
    }

    // Determine source type from existing data
    let sourceType: InputSourceType = "static";
    
    if (data.from_user) {
      sourceType = "user";
    } else if (data.from_server) {
      sourceType = "server";
    } else if (data.from_state) {
      sourceType = "state";
    } else if (data.from_files) {
      sourceType = "files";
    }

    return {
      key: data.key || "",
      sourceType,
      staticValue: data.value || "",
      from_user: data.from_user || false,
      from_server: data.from_server || false,
      from_state: data.from_state,
      from_files: data.from_files,
      optional: data.optional || false,
      type_hint: data.type_hint || ""
    };
  });

  const handleSave = () => {
    // Convert editor state back to nodeInputArgument
    const inputArg: nodeInputArgument = {
      key: editorState.key,
      optional: editorState.optional || undefined,
      type_hint: editorState.type_hint || undefined,
    };

    // Add source-specific fields based on source type
    switch (editorState.sourceType) {
      case "static":
        inputArg.value = editorState.staticValue;
        break;
      case "user":
        inputArg.from_user = true;
        break;
      case "server":
        inputArg.from_server = true;
        break;
      case "state":
        inputArg.from_state = editorState.from_state;
        break;
      case "files":
        inputArg.from_files = editorState.from_files;
        break;
    }

    onSave?.(inputArg);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className={className}>
          {children}
        </div>
      </PopoverTrigger>
      <PopoverContent className="w-auto pr-0 py-0" side="right" align="start">
        <ScrollArea className="h-[500px]">
          <div className="w-[400px] pr-5 py-4 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-semibold">Input Argument Editor</h3>
                <div className="flex items-center space-x-2">
                  <SourceTypeIndicator type={editorState.sourceType} />
                </div>
              </div>
              <Button size="sm" onClick={handleSave}>
                Save Changes
              </Button>
            </div>

            <Separator />

            {/* Basic Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <Info className="w-4 h-4 mr-2" />
                  Argument Details
                </CardTitle>
                <CardDescription className="text-xs">
                  Configure the input argument properties
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Argument Key</Label>
                  <Input 
                    value={editorState.key}
                    onChange={(e) => setEditorState(prev => ({ ...prev, key: e.target.value }))}
                    placeholder="Enter argument key (e.g., 'prompt', 'temperature')"
                    className="text-xs"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Type Hint (Optional)</Label>
                  <Input 
                    value={editorState.type_hint}
                    onChange={(e) => setEditorState(prev => ({ ...prev, type_hint: e.target.value }))}
                    placeholder="e.g., 'string', 'number', 'boolean'"
                    className="text-xs"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="optional" 
                    checked={editorState.optional} 
                    onCheckedChange={(optional) => setEditorState(prev => ({ ...prev, optional: !!optional }))}
                  />
                  <Label htmlFor="optional" className="text-xs">
                    Optional argument (can be omitted)
                  </Label>
                </div>
              </CardContent>
            </Card>

            {/* Source Configuration */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Value Source</CardTitle>
                <CardDescription className="text-xs">
                  Choose where this argument gets its value from
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <RadioGroup 
                  value={editorState.sourceType} 
                  onValueChange={(type: InputSourceType) => setEditorState(prev => ({ ...prev, sourceType: type }))} 
                  className="space-y-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="static" id="static" />
                    <Label htmlFor="static" className="text-sm">Static Value</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="user" id="user" />
                    <Label htmlFor="user" className="text-sm">User Input</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="server" id="server" />
                    <Label htmlFor="server" className="text-sm">Server Arguments</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="state" id="state" />
                    <Label htmlFor="state" className="text-sm">Toolchain State</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="files" id="files" />
                    <Label htmlFor="files" className="text-sm">File Input</Label>
                  </div>
                </RadioGroup>

                <Separator />

                {/* Source-specific configuration */}
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
                      rows={3}
                    />
                  </div>
                )}

                {editorState.sourceType === "user" && (
                  <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-md">
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      This argument will be populated from user input when the toolchain event is triggered.
                    </p>
                  </div>
                )}

                {editorState.sourceType === "server" && (
                  <div className="p-3 bg-purple-50 dark:bg-purple-950/20 rounded-md">
                    <p className="text-xs text-purple-700 dark:text-purple-300">
                      This argument will be populated from server-side arguments.
                    </p>
                  </div>
                )}

                {editorState.sourceType === "state" && (
                  <div className="space-y-2">
                    <Label className="text-xs">State Route</Label>
                    <div className="p-2 bg-muted rounded text-xs">
                      State routing configuration will be implemented in a future update.
                    </div>
                  </div>
                )}

                {editorState.sourceType === "files" && (
                  <div className="space-y-2">
                    <Label className="text-xs">File Configuration</Label>
                    <div className="p-2 bg-muted rounded text-xs">
                      File input configuration will be implemented in a future update.
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}



