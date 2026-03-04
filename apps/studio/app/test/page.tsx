"use client";
// import React from "react";
import { use, useCallback, useEffect, useRef, useState } from "react";
import { ScrollArea, ScrollAreaHorizontal } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import ToolchainSession, { ToolchainSessionMessage, toolchainStateType } from "@/hooks/toolchain-session";
import { useContextAction } from "@/app/context-provider";
import { substituteAny } from "@/types/toolchains";
import ChatBarInput from "@/components/manual_components/chat-input-bar";
import FileDropzone from "@/components/ui/file-dropzone";
import { Textarea } from "@/components/ui/textarea";
import { delay, motion } from "framer-motion";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";
import { CHAT_RENDERING_STYLE } from "@/components/markdown/configs";
import SmoothHeightDiv from "@/components/manual_components/smooth-height-div";

const test_text = `
# Heading 1
## Heading 2
### Heading 3

## Blockquotes
\`\`\`py
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Features
y = [0, 0, 1, 1]  # Labels

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42), train_test_split(X, y, test_size=0.2, random_state=42), train_test_split(X, y, test_size=0.2, random_state=42), train_test_split(X, y, test_size=0.2, random_state=42)
\`\`\`

$$P(Spam|Sender=A,Subject Line=Re:Important Meeting) = \\frac{P(Sender=A,Subject Line=Re:Important Meeting|Spam) \\cdot P(Spam)}{P(Sender=A,Subject Line=Re:Important Meeting)} = \\frac{(0.1 \\cdot 0.5) / (0.1 + 0.5)}{0.1} = 0.5$$
`


interface DocPageProps {
  params: {
    slug: string[],
  },
  searchParams: object
}

export default function TestPage() {
  const {
    userData,
  } = useContextAction();

  const toolchainWebsocket = useRef<ToolchainSession | undefined>();
  const [sessionId, setSessionId] = useState<string>();
  const [toolchainState, setToolchainState] = useState<toolchainStateType>({});
  const toolchainStateRef = useRef<toolchainStateType>({});
  const [toolchainStateCounter, setToolchainStateCounter] = useState<number>(0);
  const [expanded, setExpanded] = useState(false);
  const [markdownText, setMarkdownText] = useState<string>("");

  const startTyping = () => {
    for (let i = 0; i < test_text.length; i++) {
      setTimeout(() => {
        setMarkdownText(test_text.slice(0, i));
      }, i * 10);
    }
  }

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
    console.log("Toolchain state updated: ", toolchainState);
  }, [toolchainState]);


  const state_change_callback = useCallback((state : toolchainStateType) => {
    // console.log("State change", toolchainStateCounter, counter_value, state);
    setToolchainState(JSON.parse(JSON.stringify(state)));
  }, [setToolchainState])

  const get_state_callback = useCallback(() => {
    return toolchainState;
  }, [toolchainState]);

  const set_state_callback = (value : toolchainStateType) => {
    toolchainStateRef.current = value;
  }

  // useEffect(() => {
  //   if (toolchainWebsocket.current) {
  //     toolchainWebsocket.current.getState = get_state_callback;
  //   }
  // }, [get_state_callback]);

  // useEffect(() => {
  //   console.log("Toolchain state updated: ", toolchainState);
  // }, [toolchainStateCounter])

  const testWebsocket = () => {
    toolchainWebsocket.current = new ToolchainSession({
      onStateChange: setToolchainState,
      onMessage: (message : ToolchainSessionMessage) => {
        // console.log(message);
        if (message.toolchain_session_id !== undefined) {
          setSessionId(message.toolchain_session_id);
        }
      }
    });
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
    if (toolchainWebsocket.current && sessionId) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId,
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
    if (toolchainWebsocket.current && sessionId) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId,
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
    if (toolchainWebsocket.current && sessionId) {
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": sessionId,
          "event_node_id": "user_question_event",
          "event_parameters": {
            "model_parameters": model_params_static,
            "question": "You're wrong. It was named after Gustav Roch."
          }
        }
      });
    }
  }

  const [counter, setCounter] = useState<number>(0);

  useEffect(() => {
    if (expanded) {
      let timeouts : NodeJS.Timeout[] = [];
      for (let i = 0; i < 40; i++) {
        const timeout_get = setTimeout(() => {
          setCounter(i);
        }, i * 100);
        timeouts.push(timeout_get);
      }

      return () => {
        // cancel timeouts
        for (let t of timeouts) {
          clearTimeout(t);
        }
      }
    } else {
      setCounter(0);
    }
  }, [expanded]);

  
  return (
    <div className="w-full h-[calc(100vh)] flex flex-row justify-center">
				<ScrollArea className="w-full">
					<div className="flex flex-row justify-center pt-10">
						<div className="max-w-[85vw] md:max-w-[70vw] lg:max-w-[45vw]">
              <div className="flex flex-col space-y-2">
                <Button onClick={testWebsocket}>
                  Test websocket.
                </Button>
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
                <Button onClick={startTyping}>
                  Start Typing
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
                <div className="border-[2px] border-red-500 w-[400px] h-[500px]">
                  <Button onClick={() => setExpanded(!expanded)}>
                    Expand
                  </Button>
                  <ScrollAreaHorizontal className="p-4 border-[2px] border-teal-500">
                    <motion.div
                      className="h-[40px] bg-green-500"
                      initial={{width: 20}}
                      animate={{
                        width: expanded ? 1700 : 20
                      }}
                      transition={{ duration: 0.5 }}
                    />
                  </ScrollAreaHorizontal>
                  <SmoothHeightDiv className="border-2 border-purple-500">
                  {/* <motion.div
                    style={{ overflow: "hidden" }}
                    initial={{ height: 0 }}
                    animate={{ height: "auto" }}
                    transition={{ duration: 0.5 }}
                    exit={{ height: 0 }}
                    key={"container"}
                  > */}
                    <p className="w-[100px]">
                      {((counter !== 0)?Array(counter).fill(" Test Enabled"):"Test Disabled")}
                    </p>
                  {/* </motion.div> */}
                  </SmoothHeightDiv>
                  <MarkdownRenderer input={markdownText} finished={false} config={CHAT_RENDERING_STYLE}/>
                </div>
                <div className="w-full h-[20px] rounded-md bg-gradient-to-l from-indigo-500 from-80% ..."/>
              </div>
            <div className="h-[100px]"/>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
