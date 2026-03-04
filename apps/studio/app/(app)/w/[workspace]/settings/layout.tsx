import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Settings | QueryLake",
  description: "Manage workspace members and integrations.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
