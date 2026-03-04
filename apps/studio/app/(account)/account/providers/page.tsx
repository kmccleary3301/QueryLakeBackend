"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useContextAction } from "@/app/context-provider";
import { modifyUserExternalProviders } from "@/hooks/querylakeAPI";

export default function AccountProvidersPage() {
  const { userData, setUserData, authReviewed, loginValid } =
    useContextAction();
  const [currentProvider, setCurrentProvider] = useState("");
  const [currentKeyInput, setCurrentKeyInput] = useState("");
  const [keyAvailable, setKeyAvailable] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const providerOptions = useMemo(() => {
    return (userData?.providers ?? []).map((provider) => provider.toLowerCase());
  }, [userData?.providers]);

  useEffect(() => {
    if (!currentProvider) return;
    const hasKey = userData?.user_set_providers.includes(currentProvider) ?? false;
    setKeyAvailable(hasKey);
    setCurrentKeyInput("");
    setStatus(null);
  }, [currentProvider, userData?.user_set_providers]);

  const saveKey = () => {
    if (!userData?.auth) return;
    if (!currentProvider) {
      setStatus("Select a provider first.");
      return;
    }
    if (!currentKeyInput.trim()) {
      setStatus("Enter an API key to save.");
      return;
    }
    setSaving(true);
    setStatus(null);
    modifyUserExternalProviders({
      auth: userData.auth,
      update: { [currentProvider]: currentKeyInput },
      onFinish: (success) => {
        setSaving(false);
        setStatus(success ? "Key saved." : "Failed to save key.");
        if (success && userData) {
          setUserData({
            ...userData,
            user_set_providers: Array.from(
              new Set([...userData.user_set_providers, currentProvider])
            ),
          });
          setKeyAvailable(true);
          setCurrentKeyInput("");
        }
      },
    });
  };

  const deleteKey = () => {
    if (!userData?.auth || !currentProvider) return;
    if (!keyAvailable) return;
    const confirmed = window.confirm(
      `Delete the stored key for "${currentProvider}"?`
    );
    if (!confirmed) return;
    setDeleting(true);
    setStatus(null);
    modifyUserExternalProviders({
      auth: userData.auth,
      delete: [currentProvider],
      onFinish: (success) => {
        setDeleting(false);
        setStatus(success ? "Key deleted." : "Failed to delete key.");
        if (success && userData) {
          setUserData({
            ...userData,
            user_set_providers: userData.user_set_providers.filter(
              (provider) => provider !== currentProvider
            ),
          });
          setKeyAvailable(false);
          setCurrentKeyInput("");
        }
      },
    });
  };

  if (!authReviewed) {
    return (
      <div className="max-w-4xl space-y-6">
        <div className="h-6 w-48 rounded bg-muted" />
        <div className="h-4 w-72 rounded bg-muted" />
      </div>
    );
  }

  if (!loginValid || !userData) {
    return (
      <div className="max-w-4xl space-y-4">
        <h1 className="text-2xl font-semibold">Providers</h1>
        <p className="text-sm text-muted-foreground">
          Sign in to manage provider keys.
        </p>
        <Button asChild size="sm">
          <Link href="/auth/login">Go to login</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Providers</h1>
        <p className="text-sm text-muted-foreground">
          Manage user-scoped provider keys. These keys apply across all workspaces.
        </p>
      </div>

      <div className="rounded-lg border border-border bg-card/40 p-5 space-y-4">
        <div className="text-sm font-semibold">Provider keys</div>
        <div className="flex flex-wrap gap-3">
          <Select value={currentProvider} onValueChange={setCurrentProvider}>
            <SelectTrigger className="w-[260px]">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent>
              {providerOptions.map((provider) => (
                <SelectItem key={provider} value={provider}>
                  {provider}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            className="min-w-[320px]"
            type="password"
            placeholder={
              keyAvailable
                ? "Key is stored. Paste a new key to rotate."
                : "Enter API key"
            }
            value={currentKeyInput}
            onChange={(event) => setCurrentKeyInput(event.target.value)}
          />

          <Button
            onClick={saveKey}
            disabled={!currentProvider || !currentKeyInput.trim() || saving}
          >
            {saving ? "Saving..." : "Save"}
          </Button>
          <Button
            variant="outline"
            onClick={deleteKey}
            disabled={!currentProvider || !keyAvailable || deleting}
          >
            {deleting ? "Deleting..." : "Delete"}
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          Keys are stored server-side and are never displayed again after saving.
          Paste a new key to rotate.
        </div>

        {status ? <p className="text-xs text-muted-foreground">{status}</p> : null}
      </div>

      <div className="rounded-lg border border-dashed border-border p-5 text-sm text-muted-foreground">
        Looking for workspace-level integrations? Use{" "}
        <span className="text-foreground">Workspace → Settings → Integrations</span>{" "}
        to view workspace integration status and future webhook controls.
      </div>
    </div>
  );
}
