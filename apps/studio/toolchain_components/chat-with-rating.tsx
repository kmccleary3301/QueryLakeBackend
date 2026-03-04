"use client";
import { componentMetaDataType, configEntriesMap, inputMapping } from "@/types/toolchain-interface";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import FileDropzone from "@/components/ui/file-dropzone";
import { useToolchainContextAction } from "@/app/app/context-provider";
import uploadFiles from "@/hooks/upload-files";
// import SmoothSlider from "@/components/custom/smooth_slider";
import { retrieveValueFromObj } from "@/hooks/toolchain-session";
import { FormEvent, useEffect, useState } from "react";
import { chatEntry } from "./chat";
import { Slider } from "@/components/ui/slider";

export const METADATA : componentMetaDataType = {
  label: "File Upload",
  type: "Input",
  category: "Input",
  description: "A dropzone box for file uploads that immediately triggers on upload.",
  config: {
		"hooks": [
      "on_rating",
    ],
		"config": [
			{
				"name": "Type",
				"type": "long_string",
        "default": "Boolean" // Or "Number"
			},
      {
        "name": "Label",
        "type": "string",
        "default": "Rate this response"
      }
		],
	}
};

export default function ChatRatingButton({
	configuration,
  entriesMap,
  demo = false
  // sendEvent = () => {},
}:{
	configuration: inputMapping,
  entriesMap: configEntriesMap,
  demo?: boolean
  // sendEvent?: (event: string, event_params: {[key : string]: substituteAny}) => void
}) {
  const { 
    userData, 
    breakpoint, 
    selectedCollections
  } = useContextAction();
  const { 
    callEvent,
    storedEventArguments,
    sessionId,
    currentEvent,
    toolchainState,
    toolchainWebsocket
  } = useToolchainContextAction();

  const handleRating = async (rating : boolean | number) => {
    const fire_queue : {[key : string]: object}[]= 
    Array(Math.max(...configuration.hooks.map(hook => hook.fire_index))).fill({});


    const hasOnUploadHook = configuration.hooks.some(hook => hook.hook === 'on_upload');
    
    configuration.hooks.forEach(hook => {
      let new_args = {};

      if (hook.hook === "on_rating") {
        new_args = {
          [`${hook.target_route}`]: rating,
        }
      }
    })

    fire_queue.forEach((fire_index_bin) => {
      Object.entries(fire_index_bin).forEach(([event, event_params]) => {
        callEvent(userData?.auth || "", event, {
          ...(storedEventArguments?.current.get(event) || {}),
          ...event_params,
        })
      })
    })
  }

  const [chatMade, setChatMade] = useState(false);

	useEffect(() => {
		if (toolchainWebsocket?.current === undefined || demo) return;
    const newValue = retrieveValueFromObj(toolchainState, ["chat_history"]) as object[];
    setChatMade(newValue?.length >= 2);
	}, [toolchainState, setChatMade, toolchainWebsocket, demo]);

  return (
    <>
      {(((currentEvent === undefined && chatMade) || true)) && (
        <>
          {((entriesMap.get("Type")?.value as string) == "Number")?(
            <div>
              <Slider className="w-[200px]"defaultValue={[5]} onValueChange={(value : number[]) => {
                const actual_value = value[0];
                console.log(actual_value);
                // handleRating(actual_value);
              }} min={0} max={10} onDragEnd={() => {
                console.log("Drag end");
              }}/>
            </div>
          ):(
            <div>
              <input type="checkbox" />
            </div>
          )}
        </>
      )}
    </>
  )
}