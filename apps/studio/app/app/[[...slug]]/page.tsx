"use client";

import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { useParams } from "next/navigation";


type app_mode_type = "create" | "session" | "view" | undefined;

import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import ToolchainSession, { CallbackOrValue, ToolchainSessionMessage, toolchainStateType } from "@/hooks/toolchain-session";
import { DivisibleSection } from "../components/section-divisible";
import { useToolchainContextAction } from "../context-provider";
import { toolchain_session } from "@/types/globalTypes";
import { toast } from "sonner";
import { useContextAction } from "@/app/context-provider";
import { resolve } from "path";
import LegacyNotice from "@/components/legacy/legacy-notice";

export default function AppPage() {


  
  const [isPending, startTransition] = useTransition();
  const resolvedParams = useParams() as {
    slug: string[],
  };

  const router = useRouter(),
        pathname = usePathname(),
        search_params = useSearchParams();
  
  const app_mode_immediate = (["create", "session", "view"].indexOf(resolvedParams["slug"]?.[0]) > -1) ? resolvedParams["slug"][0] as app_mode_type : undefined;
  const appMode = useRef<app_mode_type>(app_mode_immediate);
  const [appModeState, setAppModeState] = useState(app_mode_immediate);
  const mounting = useRef(true);
  const toolchainStateRef = useRef<toolchainStateType>({});

  
  const [firstEventRan, setFirstEventRan] = useState<boolean[]>([false, false]);


  const [toolchainSelectedBySession, setToolchainSelectedBySession] = useState<string | undefined>(undefined);
  const isUnmounting = useRef(false);

  const {
    toolchainState,
    setToolchainState,
    toolchainWebsocket,
    sessionId,
    callEvent,
    setCurrentEvent,
  } = useToolchainContextAction();

  const {
    userData,
    setSelectedToolchain,
    selectedToolchainFull,
    toolchainSessions,
    setToolchainSessions,
    activeToolchainSession,
    setActiveToolchainSession,
  } = useContextAction();
  
  const setCurrentToolchainSession = useCallback((title: string) => {
    console.log("Setting toolchain session with", sessionId?.current, title, appMode.current);

    const new_session_make : toolchain_session = {
      time: toolchainSessions.has(sessionId?.current as string) ? 
            toolchainSessions.get(sessionId?.current as string)?.time as number :
            Math.floor(Date.now() / 1000),
      toolchain: selectedToolchainFull?.id as string,
      id: sessionId?.current as string,
      title: title || "Untitled Session"
    };

    setToolchainSessions((prevSessions) => {
      // Create a new Map from the previous one
      const newMap = new Map(prevSessions);
      // Update the new Map
      newMap.set(sessionId?.current as string, new_session_make);
      // Return the new Map to update the state
      return newMap;
    });
    setActiveToolchainSession(sessionId?.current);
  }, [
    sessionId,
    toolchainSessions,
    selectedToolchainFull?.id,
    setToolchainSessions,
    setActiveToolchainSession,
  ]);

  useEffect(() => {
    console.log("TITLE CHANGED, UPDATING WITH", toolchainState.title);
    // if (sessionId !== undefined) sessionId.current = searchParams?.s as string;
    if (sessionId === undefined || 
        sessionId.current === "" || 
        sessionId.current === "undefined" || 
        !toolchainState.title ||
        !toolchainSessions.has(sessionId.current) ||
        toolchainSessions.get(sessionId.current)?.title === toolchainState.title
      ) return;
    console.log("FOLLOWING THROUGH");
    setCurrentToolchainSession(toolchainState.title as string || "Untitled Session");
  }, [sessionId, setCurrentToolchainSession, toolchainSessions, toolchainState?.title]);

  useEffect(() => {
    let newFirstEventRan = firstEventRan;
    // if (sessionId !== undefined) sessionId.current = searchParams?.s as string;
    if (sessionId && sessionId.current && newFirstEventRan[0] && !newFirstEventRan[1] && !toolchainSessions.has(sessionId.current)) {
      
      console.log("FIRST EVENT RAN; PUSHING SESSION TO HISTORY");
      setCurrentToolchainSession(toolchainStateRef.current.title as string);
    }


    if (!newFirstEventRan[0] && newFirstEventRan[1]) {
      newFirstEventRan = [false, false];
      setFirstEventRan(newFirstEventRan);
    }

    if (appMode.current === "create" && newFirstEventRan[0] && newFirstEventRan[1] && sessionId?.current !== undefined) {
      console.log("FIRST EVENT RAN; PUSHING TO SESSION");
      startTransition(() => {
        window.history.pushState(null, '', `/app/session?s=${sessionId?.current}`);
      });

      setToolchainSelectedBySession(selectedToolchainFull?.id);

      setFirstEventRan([false, false]);
      if (selectedToolchainFull?.first_event_follow_up)
        callEvent(userData?.auth as string, selectedToolchainFull.first_event_follow_up, {})
    }
  }, [
    callEvent,
    firstEventRan,
    selectedToolchainFull?.first_event_follow_up,
    selectedToolchainFull?.id,
    sessionId,
    setCurrentToolchainSession,
    toolchainSessions,
    userData?.auth,
  ]);


  const STATE_UPDATE_MIN_WAIT = 25; // ms
  const updateStateTimeout = useRef<NodeJS.Timeout | undefined>(undefined);
  const last_timeout = useRef<number>(0);

  const updateState = useCallback((state: CallbackOrValue<toolchainStateType>) => {
    // console.log("Updating state", state);
    const value = (typeof state === "function") ? state(toolchainStateRef.current) : state;
    toolchainStateRef.current = value;
    const value_copied = JSON.parse(JSON.stringify(value));
    const update_timeout = Math.max(0, Date.now() - (last_timeout.current + STATE_UPDATE_MIN_WAIT));
    // const update_timeout = STATE_UPDATE_MIN_WAIT;
    if (updateStateTimeout.current !== undefined) {
      clearTimeout(updateStateTimeout.current);
    }

    // updateStateTimeout.current = setTimeout(() => {
    //   console.log("Updating state", state);
    //   last_timeout.current = Date.now();
    //   setToolchainState(value_copied);
    // }, update_timeout);

    setToolchainState(value_copied);

    
  }, [setToolchainState]);

  const onOpenSessionTC = useCallback(( create_session : boolean) => {
    if (toolchainWebsocket?.current === undefined) return;

    if (create_session) {
      setFirstEventRan([false, false]);
      setToolchainState({});
      console.log("Creating toolchain", selectedToolchainFull?.id);
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/create",
        "arguments": {
          // "toolchain_id": "test_chat_session_normal"
          "toolchain_id": selectedToolchainFull?.id,
        }
      });
    } else {
      if (toolchainWebsocket.current === undefined) {
        console.error("No session ID provided");
        router.push("/app/create");
        return;
      }
      setToolchainState({});
      console.log("Loading toolchain", sessionId?.current);
      toolchainWebsocket.current.send_message({
        "auth": userData?.auth,
        "command" : "toolchain/load",
        "arguments": {
          "session_id": sessionId?.current as string,
        }
      });
    }
    mounting.current = false;
  }, [
    router,
    selectedToolchainFull?.id,
    sessionId,
    setToolchainState,
    toolchainWebsocket,
    userData?.auth,
  ]);

  const initializeWebsocket = useCallback((onFinish: () => void) => {
    if (toolchainWebsocket === undefined || sessionId === undefined) return;
    if (toolchainWebsocket.current !== undefined) {
      if (process.env.NODE_ENV !== "production") toast("TC WS Already Open");
      onFinish();
      return;
    }
    setToolchainState({});
    if (process.env.NODE_ENV !== "production") toast("Opening TC WS");
    toolchainWebsocket.current = new ToolchainSession({
      onStateChange: (state) => {
        // toolchainStateRef.current = state;
        updateState(state);
      },
      onCallEnd: () => {
        setFirstEventRan((prevState) => [prevState[0], true]);
      },
      onMessage: (message : ToolchainSessionMessage) => {
        console.log("Message from loaded Toolchain:", message);
        if (message.toolchain_session_id !== undefined) {
          sessionId.current = message.toolchain_session_id;
        }
        if (message.toolchain_id !== undefined) {
          console.log("Toolchain ID from loaded Toolchain:", [message.toolchain_id]);
          setSelectedToolchain(message.toolchain_id);
          // setToolchainSelectedBySession(message.toolchain_id);
        }

        // if (message.error) {
        //   toast(message.error);
        // }
      },
      onSend: (message : {command?: string}) => {
        if (message.command && message.command === "toolchain/event" && appMode) {
          setFirstEventRan((prevState) => [true, prevState[1]]);
        }
      },
      onOpen: () => {
        if (process.env.NODE_ENV !== "production") toast("Websocket opened");
        onFinish();
      },
      onCurrentEventChange: (event: string | undefined) => { setCurrentEvent(event); },
      onError: (message: object) => {
        toast("An error occurred");
        console.log("Error from loaded Toolchain:", message);
        setCurrentEvent(undefined);
      },
      onClose: () => {
        if (process.env.NODE_ENV !== "production") toast("TS WC closed unexpectedly");
        console.log("Websocket closed");
        setCurrentEvent(undefined);
      }
    });
  }, [
    sessionId,
    setCurrentEvent,
    setSelectedToolchain,
    setToolchainState,
    toolchainWebsocket,
    updateState,
  ]);

  // This is a URL change monitor to refresh content.
  useEffect(() => {
    // if (isUnmounting.current) {
    //   toast("Skipping URL Change (Unmounted)");
    //   return;
    // }
    if (selectedToolchainFull === undefined) return;
    console.log("URL Change", pathname, search_params?.get("s"));
    const url_mode = pathname?.replace(/^\/app\//, "").split("/")[0] as string;
    const new_mode = (["create", "session", "view"].indexOf(url_mode) > -1) ? url_mode as app_mode_type : undefined;
    setAppModeState(new_mode);

    if (new_mode === "create") {
      setToolchainSelectedBySession(selectedToolchainFull.id);
    }

    if (new_mode === "session" && (sessionId !== undefined) && search_params?.get("s") !== sessionId?.current) {
      console.log("Session ID Change", search_params?.get("s"), sessionId?.current);
      sessionId.current = search_params?.get("s") as string;
      setActiveToolchainSession(sessionId.current);
      initializeWebsocket(() => {onOpenSessionTC(false)});
    }

    else if (new_mode === "create" && appMode.current !== "create") {
      onOpenSessionTC(true);
      setFirstEventRan([false, false]);
      setCurrentToolchainSession(toolchainState.title as string || "Untitled Session");
      setActiveToolchainSession(undefined);
    } else if (new_mode === "create") {
      initializeWebsocket(() => {onOpenSessionTC(true)});
      setActiveToolchainSession(undefined);
    }
    appMode.current = new_mode;
  }, [
    initializeWebsocket,
    onOpenSessionTC,
    pathname,
    search_params,
    selectedToolchainFull,
    sessionId,
    setActiveToolchainSession,
    setCurrentToolchainSession,
    toolchainState.title,
  ]);
  

  // useEffect(() => {
  //   return () => {
  //     console.log("Unmounting at URL:", pathname);
  //     isUnmounting.current = true;
  //     if (isUnmounting.current && toolchainWebsocket?.current !== undefined) {
  //       // toast("Disconnecting TC WS");
  //       closeWebsocket();
  //     }
  //   }
  // }, []);

  return (
    <div className="relative h-[100vh] w-full pr-0 pl-0">
      <div className="absolute left-4 right-4 top-4 z-50">
        <LegacyNotice
          title="Legacy toolchain runner"
          description="This is the legacy runner UI. Use the workspace Runs pages to browse and debug runs in the new surface."
          workspacePath="/runs"
          ctaLabel="Open workspace Runs"
        />
      </div>
      {(selectedToolchainFull !== undefined && selectedToolchainFull.display_configuration) &&
        (
        <DivisibleSection
          section={selectedToolchainFull.display_configuration}
        />
      )}
    </div>
  );
}
