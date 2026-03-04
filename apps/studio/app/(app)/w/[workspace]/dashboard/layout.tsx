import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dashboard | QueryLake",
  description: "Workspace overview, quick links, and recent activity.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
