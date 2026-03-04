"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import RuntimeModeBanner from "@/components/toolchains/runtime-mode-banner";
import { useRuntimeMode } from "@/components/toolchains/runtime-mode";
import { useContextAction } from "@/app/context-provider";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { fetchToolchainConfig } from "@/hooks/querylakeAPI";
import ToolchainSession from "@/hooks/toolchain-session";
import { ToolChain } from "@/types/toolchains";

type LogEntry = {
  id: string;
  type: "message" | "send" | "error";
  payload: string;
  timestamp: number;
};

export default function RunPage() {
  const params = useParams<{ workspace: string; runId: string }>()!;
  const router = useRouter();
  const {
    userData,
    toolchainSessions,
    refreshToolchainSessions,
    setSelectedToolchain,
    authReviewed,
    loginValid,
  } = useContextAction();
  const { mode } = useRuntimeMode();
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [pinnedLog, setPinnedLog] = useState<LogEntry | null>(null);
  const [connectionState, setConnectionState] = useState<
    "idle" | "connecting" | "connected" | "error"
  >("idle");
  const [autoRetry, setAutoRetry] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [lastRetryAt, setLastRetryAt] = useState<number | null>(null);
  const [currentEvent, setCurrentEvent] = useState<string | undefined>(
    undefined
  );
  const sessionRef = useRef<ToolchainSession | null>(null);
  const lastEventIdRef = useRef<number | null>(null);
  const retryDelayRef = useRef(1000);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [v2SessionState, setV2SessionState] = useState<{
    session_id: string;
    rev: number;
    state: Record<string, unknown>;
    files: Record<string, unknown>;
    toolchain_id: string;
  } | null>(null);
  const [v2Jobs, setV2Jobs] = useState<
    Array<{
      job_id?: string;
      node_id?: string;
      status?: string;
      request_id?: string;
      progress?: Record<string, unknown> | null;
      result_meta?: Record<string, unknown> | null;
      created_at?: string;
      updated_at?: string;
    }>
  >([]);
  const [eventNodeId, setEventNodeId] = useState("");
  const [eventInputs, setEventInputs] = useState("{\n\n}");
  const [eventError, setEventError] = useState<string | null>(null);
  const [eventSending, setEventSending] = useState(false);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [cancelingJobId, setCancelingJobId] = useState<string | null>(null);
  const [v2Toolchain, setV2Toolchain] = useState<ToolChain | null>(null);
  const [v2Events, setV2Events] = useState<
    Array<{
      rev: number;
      kind: string;
      payload: unknown;
      actor?: string;
      ts?: number;
    }>
  >([]);
  const [eventFilterKind, setEventFilterKind] = useState<string>("all");
  const [eventFilterNode, setEventFilterNode] = useState<string>("");
  const [eventPresets, setEventPresets] = useState<
    Array<{ id: string; label: string; payload: Record<string, unknown> }>
  >([]);
  const [streamRetryToken, setStreamRetryToken] = useState(0);

  useEffect(() => {
    if (!authReviewed || !loginValid || !userData?.auth) return;
    if (!toolchainSessions.has(params.runId)) {
      refreshToolchainSessions();
    }
  }, [
    authReviewed,
    loginValid,
    userData?.auth,
    toolchainSessions,
    params.runId,
    refreshToolchainSessions,
  ]);

  const run = useMemo(() => {
    return toolchainSessions.get(params.runId);
  }, [toolchainSessions, params.runId]);

  const displayTitle =
    mode === "v2"
      ? (v2SessionState?.state?.title as string | undefined) ??
        `Session ${params.runId}`
      : run?.title;
  const displayToolchain =
    mode === "v2" ? v2SessionState?.toolchain_id ?? "—" : run?.toolchain;
  const toolchainLinkTarget =
    mode === "v2" ? v2SessionState?.toolchain_id : run?.toolchain;
  const displayStarted =
    mode === "v2"
      ? "—"
      : run?.time
        ? new Date(run.time * 1000).toLocaleString()
        : "—";

  const availableEventKinds = useMemo(() => {
    const kinds = new Set<string>();
    v2Events.forEach((event) => {
      if (event.kind) kinds.add(event.kind);
    });
    return ["all", ...Array.from(kinds)];
  }, [v2Events]);

  const filteredEvents = useMemo(() => {
    return v2Events.filter((event) => {
      if (eventFilterKind !== "all" && event.kind !== eventFilterKind) {
        return false;
      }
      if (eventFilterNode.trim()) {
        const payload = event.payload as { node_id?: string };
        if (payload?.node_id !== eventFilterNode.trim()) return false;
      }
      return true;
    });
  }, [v2Events, eventFilterKind, eventFilterNode]);

  const applyNodeTemplate = useCallback((nodeId: string) => {
    if (!v2Toolchain?.nodes) return;
    const node = v2Toolchain.nodes.find((entry) => entry.id === nodeId);
    if (!node?.input_arguments) return;
    const template: Record<string, unknown> = {};
    node.input_arguments.forEach((arg) => {
      if (arg.optional) return;
      if (arg.value !== undefined) {
        template[arg.key] = arg.value;
        return;
      }
      template[arg.key] = null;
    });
    setEventInputs(JSON.stringify(template, null, 2));
  }, [v2Toolchain?.nodes]);

  useEffect(() => {
    if (!v2Toolchain?.nodes?.length) {
      setEventPresets([]);
      return;
    }
    const presets = v2Toolchain.nodes
      .filter((node) => node.id.toLowerCase().includes("user"))
      .slice(0, 5)
      .map((node) => ({
        id: node.id,
        label: `User event → ${node.id}`,
        payload: {
          user_input: "",
        },
      }));
    setEventPresets(presets);
  }, [v2Toolchain?.nodes]);

  useEffect(() => {
    const key = `ql_run_pin_${params.runId}`;
    const raw = localStorage.getItem(key);
    if (!raw) return;
    try {
      setPinnedLog(JSON.parse(raw) as LogEntry);
    } catch {
      localStorage.removeItem(key);
    }
  }, [params.runId]);

  useEffect(() => {
    const key = `ql_run_pin_${params.runId}`;
    if (!pinnedLog) {
      localStorage.removeItem(key);
      return;
    }
    localStorage.setItem(key, JSON.stringify(pinnedLog));
  }, [pinnedLog, params.runId]);

  useEffect(() => {
    if (connectionState === "connected") {
      retryDelayRef.current = 1000;
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = null;
      }
    }
  }, [connectionState]);

  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, []);

  const scheduleAutoRetry = useCallback(() => {
    if (!autoRetry) return;
    if (retryTimeoutRef.current) return;
    const delay = retryDelayRef.current;
    retryTimeoutRef.current = setTimeout(() => {
      retryTimeoutRef.current = null;
      setConnectionState("connecting");
      setStreamRetryToken((prev) => prev + 1);
      setRetryCount((prev) => prev + 1);
      setLastRetryAt(Date.now());
      retryDelayRef.current = Math.min(retryDelayRef.current * 2, 30000);
    }, delay);
  }, [autoRetry]);

  const retryStream = () => {
    setConnectionState("connecting");
    setStreamRetryToken((prev) => prev + 1);
    setRetryCount((prev) => prev + 1);
    setLastRetryAt(Date.now());
    retryDelayRef.current = 1000;
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  };

  const fetchV2Session = useCallback(async () => {
    if (!userData?.auth) return;
    const response = await fetch(`/v2/kernel/sessions/${params.runId}`, {
      headers: {
        Authorization: `Bearer ${userData.auth}`,
      },
    });
    if (!response.ok) {
      setEventError(`Failed to load session (${response.status}).`);
      return;
    }
    const payload = await response.json();
    setV2SessionState(payload);
    if (payload?.toolchain_id) {
      fetchToolchainConfig({
        auth: userData.auth,
        toolchain_id: payload.toolchain_id,
        onFinish: (result: ToolChain) => {
          setV2Toolchain(result);
        },
      });
    }
  }, [params.runId, userData?.auth]);

  const fetchV2Jobs = useCallback(async () => {
    if (!userData?.auth) return;
    setJobsLoading(true);
    const response = await fetch(`/v2/kernel/sessions/${params.runId}/jobs`, {
      headers: {
        Authorization: `Bearer ${userData.auth}`,
      },
    });
    if (response.ok) {
      const payload = await response.json();
      setV2Jobs(payload?.jobs ?? []);
    }
    setJobsLoading(false);
  }, [params.runId, userData?.auth]);

  const sendV2Event = async () => {
    if (!userData?.auth) return;
    if (!eventNodeId.trim()) {
      setEventError("Node ID is required.");
      return;
    }
    let inputs: Record<string, unknown> = {};
    try {
      inputs = eventInputs.trim() ? JSON.parse(eventInputs) : {};
    } catch (error) {
      setEventError(`Invalid JSON: ${String(error)}`);
      return;
    }
    setEventSending(true);
    setEventError(null);
    const response = await fetch(`/v2/kernel/sessions/${params.runId}/event`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${userData.auth}`,
      },
      body: JSON.stringify({
        node_id: eventNodeId,
        inputs,
        rev: v2SessionState?.rev,
      }),
    });
    if (!response.ok) {
      setEventError(`Failed to post event (${response.status}).`);
      setEventSending(false);
      return;
    }
    await fetchV2Session();
    await fetchV2Jobs();
    setEventSending(false);
  };

  useEffect(() => {
    if (mode !== "v2") return;
    if (!authReviewed || !loginValid || !userData?.auth) return;
    fetchV2Jobs();
    const interval = setInterval(() => {
      fetchV2Jobs();
    }, 8000);
    return () => clearInterval(interval);
  }, [
    mode,
    authReviewed,
    loginValid,
    userData?.auth,
    params.runId,
    streamRetryToken,
    autoRetry,
    fetchV2Jobs,
  ]);

  useEffect(() => {
    if (mode !== "v2") return;
    if (!v2Toolchain?.nodes?.length) return;
    if (eventNodeId) return;
    const firstNode = v2Toolchain.nodes[0]?.id;
    if (firstNode) {
      setEventNodeId(firstNode);
      applyNodeTemplate(firstNode);
    }
  }, [mode, v2Toolchain?.nodes, eventNodeId, applyNodeTemplate]);

  const cancelJob = async (jobId?: string) => {
    if (!jobId || !userData?.auth) return;
    setCancelingJobId(jobId);
    const response = await fetch(
      `/v2/kernel/sessions/${params.runId}/jobs/${jobId}/cancel`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${userData.auth}`,
        },
      }
    );
    setCancelingJobId(null);
    if (response.ok) {
      await fetchV2Jobs();
    } else {
      setEventError(`Failed to cancel job (${response.status}).`);
    }
  };

  useEffect(() => {
    if (mode !== "v1") {
      setConnectionState("idle");
      return;
    }
    if (!authReviewed || !loginValid || !userData?.auth) return;

    const formatPayload = (payload: unknown) => {
      if (typeof payload === "string") return payload;
      try {
        return JSON.stringify(payload, null, 2);
      } catch {
        return String(payload);
      }
    };

    const pushLog = (type: "message" | "send" | "error", payload: unknown) => {
      const entry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        type,
        payload: formatPayload(payload),
        timestamp: Date.now(),
      };
      setLogEntries((prev) => [entry, ...prev].slice(0, 200));
    };

    setLogEntries([]);
    setConnectionState("connecting");
    const session = new ToolchainSession({
      onOpen: (conn) => {
        setConnectionState("connected");
        conn.send_message({
          auth: userData.auth,
          command: "toolchain/load",
          arguments: { session_id: params.runId },
        });
      },
      onMessage: (message) => pushLog("message", message),
      onSend: (message) => pushLog("send", message),
      onError: (message) => {
        setConnectionState("error");
        pushLog("error", message);
      },
      onCurrentEventChange: (event) => setCurrentEvent(event),
      onClose: () => {
        setConnectionState("connecting");
      },
    });

    sessionRef.current = session;

    return () => {
      session.cleanup();
      sessionRef.current = null;
    };
  }, [
    mode,
    authReviewed,
    loginValid,
    userData?.auth,
    params.runId,
    streamRetryToken,
    autoRetry,
    fetchV2Session,
    fetchV2Jobs,
    scheduleAutoRetry,
  ]);

  useEffect(() => {
    if (mode !== "v2") return;
    if (!authReviewed || !loginValid || !userData?.auth) return;

    const formatPayload = (payload: unknown) => {
      if (typeof payload === "string") return payload;
      try {
        return JSON.stringify(payload, null, 2);
      } catch {
        return String(payload);
      }
    };

    const pushLog = (
      type: "message" | "send" | "error",
      payload: unknown
    ) => {
      const entry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        type,
        payload: formatPayload(payload),
        timestamp: Date.now(),
      };
      setLogEntries((prev) => [entry, ...prev].slice(0, 200));
    };

    const controller = new AbortController();
    setLogEntries([]);
    setConnectionState("connecting");

    const fetchHistory = async () => {
      const historyUrl = `/v2/kernel/sessions/${params.runId}/events`;
      const response = await fetch(historyUrl, {
        headers: {
          Authorization: `Bearer ${userData.auth}`,
        },
        signal: controller.signal,
      });
      if (!response.ok) {
        pushLog("error", {
          status: response.status,
          statusText: response.statusText,
        });
        return;
      }
      const payload = await response.json();
      const events = (payload?.events ?? []) as Array<{
        rev: number;
        kind: string;
        payload: unknown;
        actor?: string;
        correlation_id?: string;
        ts?: number;
      }>;

      if (events.length > 0) {
        setV2Events(events);
        const historyEntries = events
          .slice()
          .reverse()
          .map((event) => ({
            id: `history-${event.rev}`,
            type: "message" as const,
            payload: formatPayload({
              event: event.kind,
              data: event,
            }),
            timestamp: event.ts ? event.ts * 1000 : Date.now(),
          }));
        setLogEntries((prev) => [...historyEntries, ...prev].slice(0, 200));
        const last = events[events.length - 1];
        lastEventIdRef.current = last.rev;
      }
    };

    const stream = async () => {
      await fetchV2Session();
      await fetchV2Jobs();
      await fetchHistory();
      const baseUrl = `/v2/kernel/sessions/${params.runId}/stream`;
      const url = lastEventIdRef.current
        ? `${baseUrl}?last_event_id=${lastEventIdRef.current}`
        : baseUrl;

      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${userData.auth}`,
        },
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        setConnectionState("error");
        pushLog("error", {
          status: response.status,
          statusText: response.statusText,
        });
        scheduleAutoRetry();
        return;
      }

      setConnectionState("connected");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleEventBlock = (block: string) => {
        const lines = block.split("\n").map((line) => line.trimEnd());
        let eventType: string | null = null;
        let eventId: string | null = null;
        const dataLines: string[] = [];

        for (const line of lines) {
          if (!line) continue;
          if (line.startsWith(":")) continue; // SSE heartbeat/comment.
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
            continue;
          }
          if (line.startsWith("id:")) {
            eventId = line.slice(3).trim();
            continue;
          }
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        }

        const rawData = dataLines.join("\n");
        const trimmedData = rawData.trim();
        const normalizedEventType = eventType?.toLowerCase();

        if (!trimmedData && !eventId && !eventType) return;
        if (
          !trimmedData &&
          normalizedEventType &&
          ["ping", "heartbeat", "keepalive"].includes(normalizedEventType)
        ) {
          return;
        }
        if (!trimmedData) return;
        let parsed: unknown = rawData;
        try {
          parsed = JSON.parse(rawData);
        } catch {
          parsed = rawData;
        }

        if (eventId) {
          const idNumber = Number(eventId);
          if (!Number.isNaN(idNumber)) {
            lastEventIdRef.current = idNumber;
          }
        }

        const typed = parsed as {
          rev?: number;
          kind?: string;
          payload?: { node_id?: string };
        };
        if (typed?.payload?.node_id) {
          setCurrentEvent(typed.payload.node_id);
        }
        const rev = typed.rev;
        const kind = typed.kind;
        if (typeof rev === "number") {
          setV2SessionState((prev) =>
            prev ? { ...prev, rev } : prev
          );
          if (kind) {
            setV2Events((prev) => {
              const exists = prev.some((event) => event.rev === rev);
              if (exists) return prev;
              const next = [
                {
                  rev,
                  kind,
                  payload: typed.payload,
                },
                ...prev,
              ];
              return next.slice(0, 200);
            });
          }
        }

        pushLog("message", {
          event: eventType ?? "message",
          data: parsed,
        });
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (buffer.trim().length > 0) {
              handleEventBlock(buffer);
              buffer = "";
            }
            if (!controller.signal.aborted) {
              setConnectionState("error");
              pushLog("error", { error: "Stream ended." });
              scheduleAutoRetry();
            }
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          buffer = buffer.replace(/\r\n/g, "\n");
          let index = buffer.indexOf("\n\n");
          while (index !== -1) {
            const block = buffer.slice(0, index);
            buffer = buffer.slice(index + 2);
            if (block.trim().length > 0) {
              handleEventBlock(block);
            }
            index = buffer.indexOf("\n\n");
          }
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        setConnectionState("error");
        pushLog("error", { error: String(error) });
        scheduleAutoRetry();
      }
    };

    stream();

    return () => {
      controller.abort();
    };
  }, [
    mode,
    authReviewed,
    loginValid,
    userData?.auth,
    params.runId,
    streamRetryToken,
    autoRetry,
    fetchV2Session,
    fetchV2Jobs,
    scheduleAutoRetry,
  ]);

  const openLegacySession = () => {
    if (run?.toolchain) {
      setSelectedToolchain(run.toolchain);
    }
    router.push(`/app/session?s=${params.runId}`);
  };

  return (
    <div className="space-y-6">
      <RuntimeModeBanner />
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href={`/w/${params.workspace}`}>Workspace</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbLink href={`/w/${params.workspace}/runs`}>Runs</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>{params.runId}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <h1 className="text-2xl font-semibold">Run: {params.runId}</h1>
          <p className="text-sm text-muted-foreground">
            Workspace {params.workspace}
          </p>
        </div>
        <div className="flex gap-2">
          {mode === "v1" && (
            <Button onClick={openLegacySession} variant="outline">
              Open legacy session
            </Button>
          )}
          <Button asChild variant="outline">
            <Link href={`/w/${params.workspace}/runs`}>Back to runs</Link>
          </Button>
        </div>
      </div>

      {!authReviewed ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Loading run details...
        </div>
      ) : !loginValid || !userData ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Sign in to view run details.
        </div>
      ) : !run && mode !== "v2" ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Run not found in local history. Try refreshing or check the legacy
          session viewer.
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
            <div className="space-y-4 rounded-lg border border-border p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">Run overview</h2>
                <span className="text-xs text-muted-foreground">
                  Runtime: {mode === "v2" ? "v2 sessions" : "v1 legacy"}
                </span>
              </div>
              <div className="grid gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Title</span>
                  <span className="font-medium">{displayTitle ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Toolchain</span>
                  {toolchainLinkTarget ? (
                    <Link
                      href={`/w/${params.workspace}/toolchains/${toolchainLinkTarget}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {displayToolchain ?? toolchainLinkTarget}
                    </Link>
                  ) : (
                    <span className="font-medium">{displayToolchain ?? "—"}</span>
                  )}
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Started</span>
                  <span className="font-medium">{displayStarted}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Current event</span>
                  <span className="font-medium">
                    {currentEvent ?? "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Connection</span>
                  <span className="font-medium">{connectionState}</span>
                </div>
                {mode === "v2" && v2SessionState && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Session rev</span>
                      <span className="font-medium">{v2SessionState.rev}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Toolchain ID</span>
                      <span className="font-medium">{v2SessionState.toolchain_id}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
            <div className="rounded-lg border border-border p-5 text-sm">
              <div className="font-semibold">Runtime notes</div>
              <p className="mt-2 text-muted-foreground">
                {mode === "v2"
                  ? "v2 session streaming is active. If the SSE stream drops, use the retry button in the stream panel."
                  : "Legacy WebSocket runtime is active. Logs and state live in /app/session."}
              </p>
            </div>
          </div>

          {mode === "v2" && (
            <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
              <div className="rounded-lg border border-border p-5 space-y-4">
                <div className="text-base font-semibold">Send event</div>
                <div className="space-y-2">
                  {v2Toolchain?.nodes?.length ? (
                    <Select
                      value={eventNodeId}
                      onValueChange={(value) => {
                        setEventNodeId(value);
                        applyNodeTemplate(value);
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select node" />
                      </SelectTrigger>
                      <SelectContent>
                        {v2Toolchain.nodes.map((node) => (
                          <SelectItem key={node.id} value={node.id}>
                            {node.id}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input
                      placeholder="Node ID"
                      value={eventNodeId}
                      onChange={(event) => setEventNodeId(event.target.value)}
                    />
                  )}
                  {eventNodeId && v2Toolchain?.nodes?.length ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => applyNodeTemplate(eventNodeId)}
                    >
                      Insert input template
                    </Button>
                  ) : null}
                  <Textarea
                    className="min-h-[160px]"
                    value={eventInputs}
                    onChange={(event) => setEventInputs(event.target.value)}
                  />
                  {eventPresets.length > 0 && (
                    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                      {eventPresets.map((preset) => (
                        <Button
                          key={preset.id}
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setEventNodeId(preset.id);
                            setEventInputs(JSON.stringify(preset.payload, null, 2));
                          }}
                        >
                          {preset.label}
                        </Button>
                      ))}
                    </div>
                  )}
                  {eventError ? (
                    <div className="text-xs text-destructive">{eventError}</div>
                  ) : null}
                  <Button onClick={sendV2Event} disabled={eventSending}>
                    {eventSending ? "Sending..." : "Send event"}
                  </Button>
                </div>
              </div>
              <div className="rounded-lg border border-border p-5 space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">Jobs</div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={fetchV2Jobs}
                    disabled={jobsLoading}
                  >
                    {jobsLoading ? "Refreshing..." : "Refresh"}
                  </Button>
                </div>
                {v2Jobs.length === 0 ? (
                  <div className="text-xs text-muted-foreground">
                    No jobs recorded yet.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {v2Jobs.map((job) => (
                      <div
                        key={job.job_id ?? `${job.node_id}-${job.status}`}
                        className="rounded-md border border-border bg-background px-3 py-2 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">
                            {job.node_id ?? "Unknown node"}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-muted-foreground">
                              {job.status ?? "unknown"}
                            </span>
                            {job.job_id ? (
                              <Button
                                size="sm"
                                variant="outline"
                                disabled={cancelingJobId === job.job_id}
                                onClick={() => cancelJob(job.job_id)}
                              >
                                {cancelingJobId === job.job_id
                                  ? "Cancelling..."
                                  : "Cancel"}
                              </Button>
                            ) : null}
                          </div>
                        </div>
                        <div className="mt-1 text-muted-foreground">
                          Job ID: {job.job_id ?? "—"}
                        </div>
                        {job.progress ? (
                          <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap text-[10px] text-muted-foreground">
                            {JSON.stringify(job.progress, null, 2)}
                          </pre>
                        ) : null}
                        {job.result_meta ? (
                          <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap text-[10px] text-muted-foreground">
                            {JSON.stringify(job.result_meta, null, 2)}
                          </pre>
                        ) : null}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {mode === "v2" && v2SessionState && (
            <div className="rounded-lg border border-border p-5">
              <div className="text-base font-semibold">Session state</div>
              <div className="mt-3 grid gap-4 lg:grid-cols-2">
                <div>
                  <div className="text-xs text-muted-foreground">State</div>
                  <pre className="mt-2 max-h-64 overflow-auto rounded-md border border-border bg-background p-3 text-xs text-muted-foreground">
                    {JSON.stringify(v2SessionState.state ?? {}, null, 2)}
                  </pre>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Files</div>
                  <pre className="mt-2 max-h-64 overflow-auto rounded-md border border-border bg-background p-3 text-xs text-muted-foreground">
                    {JSON.stringify(v2SessionState.files ?? {}, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {mode === "v2" && v2Events.length > 0 && (
            <div className="rounded-lg border border-border p-5">
              <div className="flex items-center justify-between">
                <div className="text-base font-semibold">Event timeline</div>
                <span className="text-xs text-muted-foreground">
                  {filteredEvents.length} events
                </span>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Select
                  value={eventFilterKind}
                  onValueChange={(value) => setEventFilterKind(value)}
                >
                  <SelectTrigger className="w-[220px]">
                    <SelectValue placeholder="Filter by kind" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableEventKinds.map((kind) => (
                      <SelectItem key={kind} value={kind}>
                        {kind}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  className="w-[240px]"
                  placeholder="Filter by node_id"
                  value={eventFilterNode}
                  onChange={(event) => setEventFilterNode(event.target.value)}
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setEventFilterKind("all");
                    setEventFilterNode("");
                  }}
                >
                  Clear filters
                </Button>
              </div>
              <div className="mt-3 space-y-3">
                    {filteredEvents
                      .slice()
                      .sort((a, b) => b.rev - a.rev)
                      .slice(0, 50)
                      .map((event) => (
                        <div
                          key={`event-${event.rev}`}
                          className="rounded-md border border-border bg-background px-4 py-3 text-sm"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{event.kind}</span>
                            <span className="text-xs text-muted-foreground">
                              rev {event.rev}
                            </span>
                          </div>
                          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-xs text-muted-foreground">
                            {JSON.stringify(event.payload ?? {}, null, 2)}
                          </pre>
                          <div className="mt-2 flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                navigator.clipboard.writeText(
                                  JSON.stringify(event.payload ?? {}, null, 2)
                                );
                              }}
                            >
                              Copy payload
                            </Button>
                          </div>
                        </div>
                      ))}
              </div>
            </div>
          )}

          <div className="rounded-lg border border-border p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">
                Live event stream {mode === "v2" ? "(v2 SSE)" : "(v1 WS)"}
              </h2>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>Connection: {connectionState}</span>
                <span>Showing latest {logEntries.length} events</span>
                <span>Retries: {retryCount}</span>
                {lastRetryAt ? (
                  <span>Last retry: {new Date(lastRetryAt).toLocaleTimeString()}</span>
                ) : null}
                <div className="flex items-center gap-2">
                  <span>Auto-retry</span>
                  <Switch checked={autoRetry} onCheckedChange={setAutoRetry} />
                </div>
                <Button size="sm" variant="outline" onClick={retryStream}>
                  Retry stream
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setLogEntries([])}
                >
                  Clear log
                </Button>
              </div>
            </div>
            {connectionState === "error" && (
              <div className="mt-3 text-xs text-destructive">
                {mode === "v2"
                  ? "Stream disconnected. Retry the SSE connection to resume updates."
                  : "WebSocket disconnected. Retry to reconnect the live log stream."}
              </div>
            )}
            {pinnedLog && (
              <div className="mt-4 rounded-md border border-border bg-card/40 p-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-medium uppercase text-muted-foreground">
                    Pinned {pinnedLog.type}
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setPinnedLog(null)}
                  >
                    Unpin
                  </Button>
                </div>
                <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-muted-foreground">
                  {pinnedLog.payload}
                </pre>
              </div>
            )}
            <div className="mt-4 space-y-3">
              {logEntries.length === 0 ? (
                <div className="rounded-md border border-dashed border-border p-4 text-xs text-muted-foreground">
                  No events yet. If the run is active, new messages will show up
                  here.
                </div>
              ) : (
                logEntries.map((entry) => (
                  <div
                    key={entry.id}
                    className="rounded-md border border-border bg-background p-3 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium uppercase text-muted-foreground">
                        {entry.type}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setPinnedLog(entry)}
                        >
                          Pin
                        </Button>
                      </div>
                    </div>
                    <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap text-muted-foreground">
                      {entry.payload}
                    </pre>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
