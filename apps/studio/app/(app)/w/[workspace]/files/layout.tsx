import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Files | QueryLake",
  description: "Upload and manage workspace files and sources.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
