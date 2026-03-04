"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import ToolchainSession, { CallbackOrValue, ToolchainSessionMessage, toolchainStateType } from "@/hooks/toolchain-session";
import { useContextAction } from "@/app/context-provider";
import ChatBarInput from "@/components/manual_components/chat-input-bar";
import FileDropzone from "@/components/ui/file-dropzone";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { WavyCircularDisplay } from "./components/wavy-circular";
import { QueryLakeLogo, create_logo_svg } from "@/components/logo";
import "./components/logo_css.css";
import { Trash } from "lucide-react";
// import { produce } from 'immer';

const SvgComponent = ({
  className = ""
}:{
  className?: string
}) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="272.2931" height="272.2931" viewBox="0 0 272.2931 272.2931">
    <polygon fill="currentColor" points="235.7624 177.4088 272.2931 140.878 272.2931 192.5403 235.7624 229.071 235.7624 177.4088 235.7624 177.4088"/>
    <polygon fill="currentColor" points="177.4088 235.7624 229.071 235.7624 192.5403 272.2931 140.878 272.2931 177.4088 235.7624 177.4088 235.7624"/>
    <polygon fill="currentColor" points="94.8843 235.7624 131.4151 272.2931 79.7528 272.2931 43.2221 235.7624 94.8843 235.7624 94.8843 235.7624"/>
    <polygon fill="currentColor" points="36.5307 177.4088 36.5307 229.071 0 192.5403 0 140.878 36.5307 177.4088 36.5307 177.4088"/>
    <polygon fill="currentColor" points="36.5307 94.8843 0 131.4151 0 79.7528 36.5307 43.2221 36.5307 94.8843 36.5307 94.8843"/>
    <polygon fill="currentColor" points="94.8843 36.5307 43.2221 36.5307 79.7528 0 131.4151 0 94.8843 36.5307 94.8843 36.5307"/>
    <polygon fill="currentColor" points="177.4088 36.5307 140.878 0 192.5403 0 229.071 36.5307 177.4088 36.5307 177.4088 36.5307"/>
    <polygon fill="currentColor" points="235.7624 94.8843 235.7624 43.2221 272.2931 79.7528 272.2931 131.4151 235.7624 94.8843 235.7624 94.8843"/>
    <polygon fill="currentColor" points="179.0324 179.0323 179.0324 230.6946 230.6946 230.6946 230.6946 179.0323 179.0324 179.0323 179.0324 179.0323"/>
    <polygon fill="currentColor" points="136.1465 196.7962 99.6158 233.327 136.1465 269.8577 172.6773 233.327 136.1465 196.7962 136.1465 196.7962"/>
    <polygon fill="currentColor" points="93.2607 179.0323 41.5985 179.0323 41.5985 230.6946 93.2607 230.6946 93.2607 179.0323 93.2607 179.0323"/>
    <polygon fill="currentColor" points="75.4969 136.1465 38.9661 99.6158 2.4354 136.1465 38.9661 172.6773 75.4969 136.1465 75.4969 136.1465"/>
    <polygon fill="currentColor" points="93.2607 93.2607 93.2607 41.5985 41.5985 41.5985 41.5985 93.2607 93.2607 93.2607 93.2607 93.2607"/>
    <polygon fill="currentColor" points="136.1465 75.4969 172.6773 38.9661 136.1465 2.4354 99.6158 38.9661 136.1465 75.4969 136.1465 75.4969"/>
    <polygon fill="currentColor" points="179.0324 93.2607 230.6946 93.2607 230.6946 41.5985 179.0324 41.5985 179.0324 93.2607 179.0324 93.2607"/>
    <polygon fill="currentColor" points="196.7962 136.1465 233.327 172.6773 269.8577 136.1465 233.327 99.6158 196.7962 136.1465 196.7962 136.1465"/>
    <polygon fill="currentColor" points="140.0662 137.7701 191.7285 137.7701 228.2592 174.3009 176.597 174.3009 140.0662 137.7701 140.0662 137.7701"/>
    <polygon fill="currentColor" points="137.7701 140.0662 174.3009 176.597 174.3009 228.2592 137.7701 191.7285 137.7701 140.0662 137.7701 140.0662"/>
    <polygon fill="currentColor" points="134.523 140.0662 134.523 191.7285 97.9922 228.2592 97.9922 176.597 134.523 140.0662 134.523 140.0662"/>
    <polygon fill="currentColor" points="132.2269 137.7701 95.6961 174.3009 44.0339 174.3009 80.5646 137.7701 132.2269 137.7701 132.2269 137.7701"/>
    <polygon fill="currentColor" points="132.2269 134.5229 80.5646 134.5229 44.0339 97.9922 95.6961 97.9922 132.2269 134.5229 132.2269 134.5229"/>
    <polygon fill="currentColor" points="134.523 132.2269 97.9922 95.6961 97.9922 44.0339 134.523 80.5646 134.523 132.2269 134.523 132.2269"/>
    <polygon fill="currentColor" points="137.7701 132.2269 137.7701 80.5646 174.3009 44.0339 174.3009 95.6961 137.7701 132.2269 137.7701 132.2269"/>
    <polygon fill="currentColor" points="140.0662 134.5229 176.597 97.9922 228.2592 97.9922 191.7285 134.5229 140.0662 134.5229 140.0662 134.5229"/>
  </svg>
);

export default function TestPage() {
  const {
    userData,
  } = useContextAction();

  // const [toolchainWebsocket, setToolchainWebsocket] = useState<ToolchainSession | undefined>();
  const toolchainWebsocket = useRef<ToolchainSession | undefined>();
  const sessionId = useRef<string>();
  const toolchainStateRef = useRef<toolchainStateType>({});
  const [toolchainState, setToolchainState] = useState<toolchainStateType>({});
  const [toolchainStateCounter, setToolchainStateCounter] = useState<number>(0);
  const toolchainStreamMappings = useRef<Map<string, (string | number)[][]>>(new Map());
  

  const updateState = useCallback((state: CallbackOrValue<toolchainStateType>) => {
    // console.log("update state called with", toolchainStateCounter);
    setToolchainStateCounter((prev) => prev + 1);

    const value = (typeof state === "function") ? state(toolchainStateRef.current) : state;
    toolchainStateRef.current = value;
    const value_copied = JSON.parse(JSON.stringify(value));
    setToolchainState(value_copied);
    console.log(value);
  }, [setToolchainStateCounter, setToolchainState]);

  // const updateState = useCallback((state: CallbackOrValue<toolchainStateType>) => {
  //   console.log("update state called with", toolchainStateCounter);
  //   setToolchainStateCounter(toolchainStateCounter + 1);

  //   const nextState = produce(toolchainState, (draftState : toolchainStateType) => {
  //     const value = (typeof state === "function") ? state(toolchainStateRef.current) : state;
  //     return Object.assign(draftState, value);
  //   });

  //   toolchainStateRef.current = nextState;
  //   setToolchainState(nextState);
  //   console.log(nextState);
  // }, [toolchainState, setToolchainStateCounter, toolchainStateCounter, setToolchainState]);

  
  const updateStateRef = useRef(updateState);
  updateStateRef.current = updateState;
  
  useEffect(() => {
    if (toolchainWebsocket.current) { return; }

    // const ws = new ToolchainSession({
    //   onStateChange: updateState,
    //   onTitleChange: () => {},
    //   onMessage: (message : ToolchainSessionMessage) => {
    //     // console.log(message);
    //     if (message.toolchain_session_id !== undefined) {
    //       sessionId.current = message.toolchain_session_id;
    //     }
    //   }
    // });
    // toolchainWebsocket.current = ws;


    // return () => { toolchainWebsocket.current?.socket.close(); toolchainWebsocket.current = undefined;}
  }, []);

  useEffect(() => {
    console.log("Toolchain state updated: ", toolchainState);
  }, [toolchainState])


  const model_params_static = {
    "model": "mistral-7b-instruct-v0.1",
    "max_tokens": 1000, 
    "temperature": 0.1, 
    "top_p": 0.1, 
    "repetition_penalty": 1.15,
    "stop": ["<|im_end|>"],
    "include_stop_str_in_output": true
  }


  useEffect(() => {
    console.log("Session ID: ", sessionId);
  }, [sessionId]);

  useEffect(() => {
    console.log("Toolchain state updated: ", toolchainState);
  }, [toolchainState]);


  const state_change_callback = useCallback((state : toolchainStateType, counter_value : number) => {
    console.log("State change", toolchainStateCounter, counter_value, state);
    setToolchainState(state);
    setToolchainStateCounter(counter_value);
  }, [setToolchainStateCounter, toolchainStateCounter, setToolchainState])

  useEffect(() => {
    console.log("Toolchain state updated: ", toolchainState);
  }, [toolchainState])

  const testWebsocket = () => {
    // setToolchainWebsocket(new ToolchainSession({
    //   onStateChange: state_change_callback,
    //   onTitleChange: () => {},
    //   onMessage: (message : ToolchainSessionMessage) => {
    //     // console.log(message);
    //     if (message.toolchain_session_id !== undefined) {
    //       setSessionId(message.toolchain_session_id);
    //     }
    //   }
    // }));
  }
  
  const sendMessage1 = () => {
    if (toolchainWebsocket.current) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/create",
        "arguments": {
          // "toolchain_id": "test_chat_session_normal"
          "toolchain_id": "test_chat_session_normal_streaming"
        }
      });
    }
  }

  const sendMessage2 = () => {
    if (toolchainWebsocket.current) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId.current,
          "event_node_id": "user_question_event",
          "event_parameters": {
            "model_parameters": model_params_static,
            "question": "What is the Riemann-Roch theorem?"
          }
        }
      });
    }
  }

  const sendMessage3 = () => {
    if (toolchainWebsocket.current) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId.current,
          "event_node_id": "user_question_event",
          "event_parameters": {
            "model_parameters": model_params_static,
            "question": "Who are the two people the Riemann-Roch Theorem is named after?"
          }
        }
      });
    }
  }

  const sendMessage4 = () => {
    if (toolchainWebsocket.current) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId.current,
          "event_node_id": "user_question_event",
          "event_parameters": {
            "model_parameters": model_params_static,
            "question": "You're wrong. It was named after Gustav Roch."
          }
        }
      });
    }
  }


  
  return (
    <div className="w-full h-[calc(100vh)] flex flex-row justify-center">
				<ScrollArea className="w-full">
        <div className="flex flex-row justify-center pt-10">
          <div className="max-w-[85vw] md:max-w-[70vw] lg:max-w-[45vw]">
            
            <div className="flex flex-col space-y-2">
              {/* <Button onClick={testWebsocket}>
                Test websocket.
              </Button> */}
              <Button onClick={sendMessage1}>
                Send message 1.
              </Button>
              <Button onClick={sendMessage2}>
                Question 1
              </Button>
              <Button onClick={sendMessage3}>
                Question 2
              </Button>
              <Button onClick={sendMessage4}>
                Question 3
              </Button>
              <Button onClick={create_logo_svg}>
                Create SVG Logo
              </Button>

              <ChatBarInput/>

              <FileDropzone onFile={(file) => console.log(file)} />

              <ScrollArea className="w-auto h-[200px] rounded-md border-[2px] border-secondary">
                <Textarea 
                  className="w-full h-full scrollbar-hide"
                  value={JSON.stringify(toolchainState, null, "\t")} 
                  onChange={() => {}}
                />
                {/* <p>{JSON.stringify(toolchainState, null, "\t")}</p> */}
              </ScrollArea>
              <WavyCircularDisplay 
                className="w-[400px] h-[400px] mx-0"
                containerClassName="transform-gpu w-[400px] h-[400px]"
                canvasClassName="w-[400px] h-[400px] blur-[0px]"
                blur={0}
                waveWidth={4} 
                waveCount={4} 
                waveAmplitude={0.1}
                wavePinchEnd={0}
                wavePinchMiddle={0.064}
                speed={14}
                backgroundFill="primary"
              >
                <div className="w-400 h-400"/>
              </WavyCircularDisplay>
              <QueryLakeLogo 
                className="w-[400px] h-[400px] mx-0"
                containerClassName="transform-gpu w-[400px] h-[400px]"
                canvasClassName="w-[400px] h-[400px] blur-[0px]"
                blur={0}
                waveWidth={4} 
                waveCount={8} 
                waveAmplitude={0.8}
                wavePinchEnd={0}
                wavePinchMiddle={0.064}
                speed={14}
                backgroundFill="primary"
              >
                <div className="w-400 h-400"/>
              </QueryLakeLogo>
              <SvgComponent/>
              <Trash/>
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
