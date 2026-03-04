import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Collections | QueryLake",
  description: "Manage and organize workspace collections.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
