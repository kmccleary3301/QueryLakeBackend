import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Platform | QueryLake",
  description: "Manage API keys, usage, and platform controls.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
