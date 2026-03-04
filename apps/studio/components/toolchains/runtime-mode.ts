"use client";

import { useEffect, useState } from "react";

export type RuntimeMode = "v1" | "v2";

const STORAGE_KEY = "ql_toolchain_runtime_mode";

export const readRuntimeMode = (): RuntimeMode => {
  if (typeof window === "undefined") return "v1";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "v2" ? "v2" : "v1";
};

export const storeRuntimeMode = (mode: RuntimeMode) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, mode);
};

export const useRuntimeMode = () => {
  const [mode, setMode] = useState<RuntimeMode>("v1");

  useEffect(() => {
    setMode(readRuntimeMode());
  }, []);

  const updateMode = (next: RuntimeMode) => {
    setMode(next);
    storeRuntimeMode(next);
  };

  return { mode, setMode: updateMode };
};
