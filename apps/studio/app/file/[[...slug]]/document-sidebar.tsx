import * as React from "react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
} from "@/components/ui/sidebar"
import { ComboBox, ComboBoxScrollPreview } from "@/components/ui/combo-box"
import { userDataType } from "@/types/globalTypes"
import { Button } from "@/components/ui/button"
import { createCollection, fetch_collection_document_type, modifyCollection, QueryLakeChangeCollectionOwnership } from "@/hooks/querylakeAPI"
import { toast } from "sonner"
import CompactInput from "@/components/ui/compact-input"
import { Textarea } from "@/components/ui/textarea"
import { useContextAction } from "@/app/context-provider"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { ColumnSchema } from "./columns"
import axios from "axios"
import craftUrl from "@/hooks/craftUrl"
import { useParams, useRouter } from "next/navigation"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { fontConsolas } from "@/lib/fonts"

type uploading_file_type = {
  title: string,
  progress: number,
}

type collection_mode_type = "create" | "edit" | "view" | undefined;

export function CollectionSidebar({
	document_id,
	user_auth,
	collection_name,
	set_collection_name,
	...props 
}:React.ComponentProps<typeof Sidebar> & {
	user_auth: userDataType,
	document_id: string,
	collection_name: string,
	set_collection_name: React.Dispatch<React.SetStateAction<string>>,
}) {

	const router = useRouter();
	const params = useParams() as {
    slug: string[],
  };

	const collection_mode_immediate = 
  (
    ["create", "edit", "view"].indexOf(
    params["slug"][0]) 
    > -1
  ) ? 
    params["slug"][0] as collection_mode_type :
    undefined;

	const { refreshCollectionGroups } = useContextAction();

	const [tempName, setTempName] = React.useState(collection_name);


	
	
	const all_available_orgs = [
		{category_label: "Self", values: [
			{value: "personal", label: "Personal", preview: "Collection belongs to you."},
			...(user_auth.is_admin)?
				[{value: "global", label: "Global", preview: "Collection is viewable by everyone."}] :
				[],
		]},
		{category_label: "Organizations", values: user_auth.memberships.filter((membership) => {
			return (membership.role === "owner") || (membership.role === "admin");
		}).map((membership) => {
			return {
				value: membership.organization_id, 
				label: membership.organization_name, 
				preview: "Collection belongs to " + membership.organization_name + "."
			}
		})}
	];


  return (
    <Sidebar {...props}>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Modify Collection</SidebarGroupLabel>
          <SidebarGroupContent className="space-y-4 flex flex-col">
						
						<div className="flex flex-col space-y-2">
							<Label>ID</Label>
							<Input
								className="h-6 text-sm px-1"
								readOnly
								value={document_id}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Label>File Name</Label>
							<Textarea
								readOnly
								value={tempName}
								placeholder="Collection Name"
								onChange={(e) => {
									setTempName(e.target.value);
								}}
							/>
						</div>
						
						{/* <div className="flex flex-col space-y-2">
							<Label>Description</Label>
							<Textarea
								readOnly
								value={tempDescription}
								placeholder="Collection Description"
								className="resize-none"
								rows={10}
								onChange={(e) => {
									setTempDescription(e.target.value);
								}}
							/>
						</div> */}

          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
			<SidebarFooter>
				
			</SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
