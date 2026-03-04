"use client";
import { Skeleton } from "@/components/ui/skeleton";
import { componentMetaDataType, inputMapping } from "@/types/toolchain-interface";
import tailwindToObject from "@/hooks/tailwind-to-obj/tailwind-to-style-obj-imported";
import { useContextAction } from "@/app/context-provider";
import ToolchainSession, { retrieveValueFromObj } from "@/hooks/toolchain-session";
import {default as ChatInputProto} from "@/components/ui/chat-input";
import { substituteAny } from "@/types/toolchains";
import { useToolchainContextAction } from "@/app/app/context-provider";
import craftUrl from "@/hooks/craftUrl";
import uploadFiles from "@/hooks/upload-files";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { AlarmSmoke, Cylinder, DollarSign, Droplet, User, Wind } from "lucide-react";
import { motion } from "framer-motion";

export const METADATA : componentMetaDataType = {
  label: "BASF Intro Screen",
  type: "Input",
  category: "Text",
  description: "A Preview for chat intro.",
  config: {
		"hooks": [
      "on_upload", 
      "on_submit",
      "selected_collections",
    ],
    "config": [
      {
				"name": "test_7_long_string",
				"type": "long_string",
        "default": "Hello, how are you?"
			}
		],
	}
};


const test_prompts = [
	{
		icon: () => (
			<AlarmSmoke className="w-6 h-6 text-yellow-400" style={{
				color: "#facc15"
			}} />
		),
    header: "How do we handle vapor recovery?",
		prompt: "How do we handle vapor recovery?",
	},
	{
		icon: () => (
			<Droplet className="w-6 h-6 text-green-400" style={{
				color: "#4ade80"
			}} />
		),
    
		header: "Tell me about mineral oil procedures.",
    prompt: "What standards are there regarding mineral insulating oil?"
	},
	{
		icon: () => (
			<Cylinder className="w-6 h-6 text-red-400" style={{
				color: "#f87171"
			}} />
		),
		header: "What can you find on boilers?",
    prompt: "How are we supposed to handle boiler pressure?"
	},
	{
		icon: () => (
			<Wind className="w-6 h-6 text-purple-500" />
		),
		header: "Flue-gas analysis guides.",
    prompt: "Tell me about flue gas analysis."
	}
]

export default function BasfIntroScreen({
	configuration, //Hello, , , ,, 
	demo = false,
}:{
	configuration: inputMapping,
	demo?: boolean,
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
		toolchainState,
		toolchainWebsocket,
  } = useToolchainContextAction();

	const [currentValue, setCurrentValue] = useState(
    retrieveValueFromObj(toolchainState, ["chat_history"])
	);
	
	const [showTitle, setShowTitle] = useState(false);
	const [showButtons, setShowButtons] = useState(false);
  const [show, setShow] = useState(false);


	useEffect(() => {
		if (toolchainWebsocket?.current === undefined) return;
    const newValue = retrieveValueFromObj(toolchainState, ["chat_history"]);
    const newShow = (newValue === undefined || (Array.isArray(newValue) && newValue.length === 0))
    setShow(newShow);
    // if (JSON.stringify(newValue) !== JSON.stringify(currentValue)) {
    //   setCurrentValue(newValue);
    // }
	}, [toolchainState]);

  useEffect(() => {
    if (currentValue === undefined || (Array.isArray(currentValue) && currentValue.length === 0) || demo) {
      setTimeout(() => setShowTitle(true), 100);
      setTimeout(() => setShowButtons(true), 600);
    }
  }, [currentValue, demo]);

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

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
		<>
			{(show || demo || true) && (
				<div className="w-auto h-[calc(100vh-40px)] flex flex-col justify-center pt-[30px]" style={{
					height: "calc(100vh - 180px)",
					paddingTop: "140px",
				}}>
				<div style={tailwindToObject([configuration.tailwind], breakpoint)}>
					<span className="w-auto text-center flex flex-col gap-8">
						<motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-3xl font-soehne"
              >
                <strong>What can I help you find?</strong>
              </motion.h1>
              <motion.div 
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className="w-auto flex flex-wrap justify-center gap-4"
              >
                {test_prompts.map((prompt, index) => (
                  <motion.div
                    key={index}
                    variants={itemVariants}
                    transition={{ duration: 0.5 }}
                  >
                    <Button
                      onClick={() => handleSubmission(prompt.prompt, [])}
                      variant={"ghost"}
                      className="text-base flex flex-col justify-start space-y-2 items-center overflow-wrap w-[140px] h-[120px] rounded-xl"
                      style={{ width: "140px" }}
                    >
                      <span className="w-auto text-center">{prompt.icon()}</span>
                      <span className="w-auto text-sm text-wrap text-center">{prompt.header}</span>
                    </Button>
                  </motion.div>
                ))}
              </motion.div>
					</span>
				</div>
				</div>
			)}
		</>
  )
}