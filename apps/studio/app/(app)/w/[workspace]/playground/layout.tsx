import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Playground | QueryLake",
  description: "Experiment with retrieval, prompts, and model calls.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
