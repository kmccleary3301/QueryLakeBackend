"use client";
import { Skeleton } from "@/components/ui/skeleton";
import { componentMetaDataType, inputMapping } from "@/types/toolchain-interface";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import ToolchainSession from "@/hooks/toolchain-session";
import {default as ChatInputProto} from "@/components/ui/chat-input";
import { substituteAny } from "@/types/toolchains";
import { useToolchainContextAction } from "@/app/app/context-provider";
import craftUrl from "@/hooks/craftUrl";
import uploadFiles from "@/hooks/upload-files";

export const METADATA : componentMetaDataType = {
  label: "Chat Input",
  type: "Input",
  category: "Text",
  description: "A chat input component that optionally supports file uploads.",
  config: {
		"hooks": [
      "on_upload", 
      "on_submit",
      "selected_collections",
    ]
	}
};

export default function Component({
	configuration, //Hello, , , ,, 
}:{
	configuration: inputMapping,
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
  } = useToolchainContextAction();

  const handleSubmission = async (text : string, files: File[]) => {
    const fire_queue : {[key : string]: object}[]= 
    Array(Math.max(...configuration.hooks.map(hook => hook.fire_index))).fill({});
    const collections = Array.from(selectedCollections.keys()).filter((key) => selectedCollections.get(key) === true);
    console.log("ChatInput handleSubmission", text, files, collections, Array.from(selectedCollections.entries()));
    
    const hasOnUploadHook = configuration.hooks.some(hook => hook.hook === 'on_upload');
    let file_upload_hashes : {type: "<<||TOOLCHAIN_SESSION_FILE||>>", document_hash_id: string, name: string}[] = [];
    if (hasOnUploadHook && files.length > 0) {
      const uploadFileResponses = await uploadFiles({
        files: files,
        url: "/upload/",
        parameters: {
          "auth": userData?.auth as string,
          "collection_hash_id": sessionId?.current as string,
          "collection_type" : "toolchain_session",
          "await_embedding": true,
        }
      });
      file_upload_hashes = uploadFileResponses.map((response : any) => response.hash_id)
                                              .filter((hash : string) => hash !== undefined)
                                              .map((hash : string) => ({
                                                type: "<<||TOOLCHAIN_SESSION_FILE||>>",
                                                document_hash_id: hash,
                                                name: "hi" //TODO: Change this to the actual file name
                                              }));
    }

    configuration.hooks.forEach(hook => {
      let new_args = {};

      if (hook.hook === "on_submit") {
        new_args = {
          [`${hook.target_route}`]: text,
        }
      } else if (hook.hook === "on_upload") {
        new_args = {
          [`${hook.target_route}`]: file_upload_hashes,
        }
      } else if (hook.hook === "selected_collections") {
        new_args = {
          [`${hook.target_route}`]: collections,
        }
      }
      fire_queue[hook.fire_index-1][hook.target_event] = {
        ...fire_queue[hook.fire_index-1][hook.target_event] || {},
        ...new_args,
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

  return (
    <ChatInputProto 
      style={tailwindToObject([configuration.tailwind], breakpoint)}
      onSubmission={handleSubmission}
    />
  )
}