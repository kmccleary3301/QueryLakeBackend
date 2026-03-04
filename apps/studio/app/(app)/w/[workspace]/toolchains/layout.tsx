import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Toolchains | QueryLake",
  description: "Inspect and run QueryLake toolchains.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
