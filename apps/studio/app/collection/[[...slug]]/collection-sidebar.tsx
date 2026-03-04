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
	collection_id,
	user_auth,
	collection_is_public,
	set_collection_is_public,
	collection_owner,
	set_collection_owner,
	collection_name,
	set_collection_name,
	collection_description,
	set_collection_description,
	add_files,
	...props 
}:React.ComponentProps<typeof Sidebar> & {
	user_auth: userDataType,
	collection_id: string,
	collection_is_public: boolean,
	set_collection_is_public: React.Dispatch<React.SetStateAction<boolean>>,
	collection_owner: string,
	set_collection_owner: React.Dispatch<React.SetStateAction<string>>,
	collection_name: string,
	set_collection_name: React.Dispatch<React.SetStateAction<string>>,
	collection_description: string,
	add_files: (files: ColumnSchema[]) => void,
	set_collection_description: React.Dispatch<React.SetStateAction<string>>,
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

	const [tempOwner, setTempOwner] = React.useState(collection_owner);
	const [tempPublic, setTempPublic] = React.useState(collection_is_public);
	const [tempName, setTempName] = React.useState(collection_name);
	const [tempDescription, setTempDescription] = React.useState(collection_description);

	const [uploadingFiles, setUploadingFiles] = React.useState<uploading_file_type[]>([]);

	const start_document_uploads = async (collection_hash_id : string, upload_files : File[]) => {
		// setPendingUploadFiles(upload_files);

    const url_2 = craftUrl("/upload/", {
      "auth": user_auth?.auth as string,
      "collection_hash_id": collection_hash_id
    });
  
    // setPublishStarted(true);
    
    setUploadingFiles(upload_files.map((file) => {
      return {
        title: file.name,
        progress: 0
      }
    }));

    const totalCount = upload_files.length;

    for (let i = 0; i < upload_files.length; i++) {
      const file = upload_files[i];
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await axios.post(url_2.toString(), formData, {
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded / (progressEvent.total || 1)) * 100);
            console.log(`File ${file.name} upload progress: ${progress}%`, progress);
            setUploadingFiles(files => [{...files[0], progress: progress}, ...files.slice(1)]);
          },
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        const response_data = response.data as {success: false} | {success : true, result: fetch_collection_document_type};

        if (response_data.success) {
          setUploadingFiles(files => files.slice(1));
          if (i === totalCount - 1) {
            // add_files(uploadedFiles);
						toast("Uploaded files successfully.");
          }
        }
        console.log("Pending Uploading Files", upload_files.length);

      } catch (error) {
        console.error(`Failed to upload file ${file.name}:`, error);
				toast(`Failed to upload file ${file.name}`);
      }
    }
  };

	const onPublish = React.useCallback(() => {
    const create_args = {
      auth: user_auth?.auth as string,
      title: tempName,
      description: tempDescription,
      public: tempPublic,
    }

    if (collection_mode_immediate === "create") {
      createCollection({
        ...create_args, 
        ...(tempOwner === "personal") ?
						{} :
						{organization_id: tempOwner},
        onFinish: (result : false | {hash_id : string}) => {
          if (result !== false) {
						router.push(`/collection/edit/${result.hash_id}`);
          } else {
						toast("Failed to create collection.");
					}
        }
      });
    }
  }, [tempName, tempDescription, tempPublic, tempOwner, collection_mode_immediate, router, user_auth?.auth]);

	React.useEffect(() => {
		setTempOwner(collection_owner);
		setTempPublic(collection_is_public);
		setTempName(collection_name);
		setTempDescription(collection_description);
	}, [collection_owner, collection_is_public, collection_name, collection_description]);

	const saveChangesOwnership = React.useCallback(() => {
		QueryLakeChangeCollectionOwnership({
			auth: user_auth?.auth as string,
			username: user_auth.username,
			owner: tempOwner,
			public: tempPublic,
			collection_id: collection_id,
			onFinish: (result : boolean) => {
				if (result) {
					set_collection_owner(tempOwner);
					set_collection_is_public(tempPublic);
					toast("Changed owner/public successfully.");
				} else {
					toast("Failed to save changes.");
				}
			}
		})
	}, [tempPublic, tempOwner, collection_id, set_collection_owner, set_collection_is_public, user_auth?.auth, user_auth.username]);

	const saveChangesMetadata = React.useCallback(() => {
		modifyCollection({
			auth: user_auth?.auth as string,
			public: collection_is_public,
			collection_id: collection_id,
			description: tempDescription,
			title: tempName,
			onFinish: (result) => {

			}
		})
	}, [collection_is_public, tempName, tempDescription, collection_id, user_auth?.auth]);

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
							<Label>Owner</Label>
							<ComboBoxScrollPreview
								value={tempOwner} 
								onChange={(value: string) => {
									setTempOwner(value);
								}}
								values={all_available_orgs}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Label>Visibility</Label>
							<ComboBox
								value={(tempPublic) ? "public" : "private"} 
								onChange={(value_change: string) => {
									if (value_change === "public") {
										setTempPublic(true);
									} else {
										setTempPublic(false);
									}
								}}
								values={[
									{value: "private", label: "Private", preview: "Collection is private, only people with access can view it."},
									{value: "public", label: "Public", preview: "Collection is viewable by everyone."},
								]}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Label>Title</Label>
							<Input
								readOnly={!(collection_mode_immediate === "edit" || collection_mode_immediate === "create")}
								value={tempName}
								placeholder="Collection Name"
								onChange={(e) => {
									setTempName(e.target.value);
								}}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Label>ID</Label>
							<Input
								className="h-6 text-sm px-1"
								readOnly
								value={collection_id}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Label>Description</Label>
							<Textarea
								readOnly={!(collection_mode_immediate === "edit" || collection_mode_immediate === "create")}
								value={tempDescription}
								placeholder="Collection Description"
								className="resize-none"
								rows={10}
								onChange={(e) => {
									setTempDescription(e.target.value);
								}}
							/>
						</div>
						<div className="flex flex-col space-y-2">
							<Input 
								id="document" 
								className='items-center text-center pb-0' 
								type="file"
								readOnly={(collection_mode_immediate !== "edit") || (uploadingFiles.length > 0)}
								multiple 
								onChange={(event) =>{
									if (event.target.files !== null) {
										// setPendingUploadFiles(Array.from(event.target.files));
										start_document_uploads(collection_id, Array.from(event.target.files));
									}
							}}/>
							{/* {pendingUploadFiles?.map((file, index) => (
								<p className={cn("text-xs overflow-hidden text-ellipsis w-full", fontConsolas.className)}>
									{file.name}
								</p>
							))} */}
							{uploadingFiles?.map((file, index) => (
								<div key={index} className="flex flex-col gap-y-1">
									<p className={cn("text-xs overflow-hidden text-ellipsis w-full", fontConsolas.className)}>
										{file.title}
									</p>
									<Progress value={file.progress} className="w-full h-2"/>
								</div>
							))}
						</div>

          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
			<SidebarFooter>
				<Button
					onClick={() => {
						if (collection_mode_immediate === "create") {
							onPublish();
						}

						if (collection_mode_immediate === "edit") {
							if ((tempOwner !== collection_owner) || (tempPublic !== collection_is_public)) {
								saveChangesOwnership();
							}
							if ((tempName !== collection_name) || (tempDescription !== collection_description)) {
								saveChangesMetadata();
							}
							refreshCollectionGroups();
						}
					}} 
					disabled={
						!(
						((collection_mode_immediate === "create") &&
						(tempName.length > 0)) || // Enable save button when creating a new collection
						
						((collection_mode_immediate === "edit") && // Enable save button when editing and something changed.
							((tempOwner === collection_owner) ||
							(tempPublic === collection_is_public) ||
							(tempName === collection_name) ||
							(tempDescription === collection_description))
						))
					}
				>
					{(collection_mode_immediate === "create")?"Publish":"Save"}
				</Button>
			</SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
