import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Status | QueryLake",
  description: "System status and runtime health.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
