"use client";

import { componentMetaDataType, configEntriesMap, inputMapping } from "@/types/toolchain-interface";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

export const METADATA : componentMetaDataType = {
  label: "Switch",
  type: "Input",
  category: "ShadCN Components",
  description: "A simple toggle switch.",
  config: {
    hooks: [
      "value_map",
    ],
    config: [
      {
        "name": "Label",
        "type": "string",
        "default": ""
      }
    ],
  }
};

export default function SwitchInput({
	configuration,
  entriesMap,
  demo = false
}:{
	configuration: inputMapping,
  entriesMap: configEntriesMap,
  demo?: boolean
}) {
  const {
    breakpoint,
  } = useContextAction();

  return (
    <div style={tailwindToObject([configuration.tailwind], breakpoint)}>
      <div className="inline-flex flex-row flex-shrink">
        <Switch className="" onCheckedChange={(c : boolean) => {}}/>
        <div className="h-auto flex flex-col justify-center pl-2">
          <Label>{entriesMap.get("Label")?.value as string}</Label>
        </div>
      </div>
    </div>
  )
}