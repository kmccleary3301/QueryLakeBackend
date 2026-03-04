"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RuntimeMode, useRuntimeMode } from "@/components/toolchains/runtime-mode";

const modeLabels: Record<RuntimeMode, string> = {
  v1: "v1 (WebSocket legacy)",
  v2: "v2 (Sessions + SSE)",
};

const modeDescriptions: Record<RuntimeMode, string> = {
  v1: "Matches the current WebSocket runtime used across the legacy UI.",
  v2: "Targets the new session system with durable logs and SSE streaming.",
};

export default function RuntimeModeBanner() {
  const { mode, setMode } = useRuntimeMode();

  return (
    <div className="rounded-lg border border-border bg-card/40 p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="text-sm font-semibold">Toolchain runtime mode</div>
          <div className="text-xs text-muted-foreground">
            Toggle which backend runtime we will target as we wire Toolchains
            into the new workspace UI.
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={mode === "v1" ? "default" : "outline"}
            onClick={() => setMode("v1")}
          >
            v1 legacy
          </Button>
          <Button
            size="sm"
            variant={mode === "v2" ? "default" : "outline"}
            onClick={() => setMode("v2")}
          >
            v2 sessions
          </Button>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline">{modeLabels[mode]}</Badge>
        <span>{modeDescriptions[mode]}</span>
      </div>
      <div className="mt-2 text-xs text-muted-foreground">
        Note: this switch only affects the frontend wiring once we connect API
        calls. It does not change backend runtime behavior yet.
      </div>
    </div>
  );
}
