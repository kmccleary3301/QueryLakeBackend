'use client';
import {
  Dispatch,
  MutableRefObject,
	PropsWithChildren,
	SetStateAction,
	createContext,
	useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import ToolchainSession, { toolchainStateType } from '@/hooks/toolchain-session';
import { substituteAny } from '@/types/toolchains';
import { usePathname } from 'next/navigation';
import { toast } from 'sonner';

const ToolchainContext = createContext<{
  toolchainStateCounter: number;
  setToolchainStateCounter: Dispatch<SetStateAction<number>>;
	toolchainState: toolchainStateType,
	setToolchainState: Dispatch<SetStateAction<toolchainStateType>>;
  toolchainWebsocket: MutableRefObject<ToolchainSession | undefined> | undefined;
  sessionId: MutableRefObject<string> | undefined;
  callEvent: (auth: string, event: string, event_params: {[key : string]: substituteAny}) => void;
  storedEventArguments: MutableRefObject<Map<string, {[key : string]: substituteAny}>> | undefined;
  currentEvent: string | undefined;
  setCurrentEvent: Dispatch<SetStateAction<string | undefined>>;
}>({
  toolchainStateCounter: 0,
  setToolchainStateCounter: () => {},
	toolchainState: {},
  setToolchainState: () => {},
  toolchainWebsocket: undefined,
  sessionId: undefined,
  callEvent: () => {},
  storedEventArguments: undefined,
  currentEvent: undefined,
  setCurrentEvent: () => {},
});

export const ToolchainContextProvider = ({
	children,
}: PropsWithChildren<{}>) => {
  const pathname = usePathname();

  const [toolchain_state_counter, set_toolchain_state_counter] = useState<number>(0);
	const [toolchain_state, set_toolchain_state] = useState<toolchainStateType>({});
  const toolchain_websocket = useRef<ToolchainSession | undefined>();
  const session_id = useRef<string>("");
  const stored_event_arguments = useRef<Map<string, {[key : string]: substituteAny}>>(new Map());
  const [current_event, set_current_event] = useState<string | undefined>();

  const call_event = (
    auth: string,
    event: string, 
    event_params: {[key : string]: substituteAny}
  ) => {
    if (toolchain_websocket?.current && session_id?.current) {
      console.log("Sending Event", event, event_params);
      toolchain_websocket.current.send_message({
        "auth": auth,
        "command" : "toolchain/event",
        "arguments": {
          "session_id": session_id.current,
          "event_node_id": event,
          "event_parameters": event_params
        }
      });
    }
  };

  const closeWebsocket = () => {
    if (toolchain_websocket?.current === undefined || toolchain_websocket.current.socket === undefined) return;
    if (process.env.NODE_ENV !== "production") toast("Closing TC WS");
    toolchain_websocket.current.cleanup();
    toolchain_websocket.current = undefined;
  }

  useEffect(() => {
    return () => {

      console.log("CONTEXT PROVIDER UNMOUNTING (TC WEBSOCKET)");
      if (toolchain_websocket?.current !== undefined) {
        // toast("Disconnecting TC WS");
        closeWebsocket();
      }
    }
  }, []);

	return (
		<ToolchainContext.Provider value={{ 
      toolchainStateCounter: toolchain_state_counter,
      setToolchainStateCounter: set_toolchain_state_counter,
			toolchainState: toolchain_state,
      setToolchainState: set_toolchain_state,
      toolchainWebsocket: toolchain_websocket,
      sessionId: session_id,
      callEvent: call_event,
      storedEventArguments: stored_event_arguments,
      currentEvent: current_event,
      setCurrentEvent: set_current_event,
		}}>
			{children}
		</ToolchainContext.Provider>
	);
};

export const useToolchainContextAction = () => {
	return useContext(ToolchainContext);
};