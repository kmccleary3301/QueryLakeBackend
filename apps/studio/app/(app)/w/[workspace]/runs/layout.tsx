import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Runs | QueryLake",
  description: "Track toolchain runs and session activity.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
