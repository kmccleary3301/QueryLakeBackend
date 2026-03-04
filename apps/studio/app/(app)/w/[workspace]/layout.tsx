import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Workspace | QueryLake",
  description: "Workspace overview for collections, toolchains, and runs.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
