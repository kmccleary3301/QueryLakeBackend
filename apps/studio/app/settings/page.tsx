"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LegacyUserSettingsRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/account/preferences");
  }, [router]);

  return (
    <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
      Redirecting...
    </div>
  );
}

