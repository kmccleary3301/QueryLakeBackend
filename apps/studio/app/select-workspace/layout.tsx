import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Select Workspace | QueryLake",
  description: "Choose your workspace to continue.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
