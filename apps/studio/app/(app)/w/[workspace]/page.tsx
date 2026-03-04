import { redirect } from "next/navigation";

type WorkspacePageProps = {
  params: Promise<{ workspace: string }>;
};

export default async function WorkspacePage({ params }: WorkspacePageProps) {
  const { workspace } = await params;
  redirect(`/w/${workspace}/dashboard`);
}
