"use client";


type collection_mode_type = "create" | "edit" | "view" | undefined;

/**
 * v0 by Vercel.
 * @see https://v0.dev/t/n2FrFZXZwwu
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */

import axios from 'axios';
import { useCallback, useEffect, useMemo, useState } from "react"
import { Label } from '@/components/ui/label';
import { SelectValue, SelectTrigger, SelectItem, SelectContent, Select } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SVGProps } from "react"
import { 
  fetchCollection, 
  fetch_collection_document_type,
  deleteDocument,
  createCollection,
  openDocument,
  modifyCollection,
  fetchCollectionDocuments
} from "@/hooks/querylakeAPI";
import { useContextAction } from "@/app/context-provider";
import craftUrl from "@/hooks/craftUrl";
import { useParams, useRouter } from 'next/navigation';
import { Progress } from '@/components/ui/progress';
import { Copy, LucideLoader2 } from 'lucide-react';
import "./spin.css";
import { handleCopy } from '@/components/markdown/markdown-code-block';

const file_size_as_string = (size : number) => {
  if (size < 1024) {
    return `${size} B`;
  } else if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  } else if (size < 1024 * 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  } else {
    return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
}

function FileDisplayType({ 
  name,
  finishedProcessing,
  subtext = undefined,
  progress = undefined,
  onOpen = undefined,
  onDelete = undefined
} : { 
  name: string,
  finishedProcessing: boolean,
  subtext?: string[],
  progress?: number,
  onOpen?: () => void,
  onDelete?: () => void
}) {
  return (
    <div className="w-[inherit] h-14 flex flex-row items-center justify-between pt-2 pb-2">
      <div className='w-[70%] pb-2 flex flex-col space-y-1'>
        {(onOpen === undefined)?(
          <p className="font-medium whitespace-nowrap overflow-hidden text-ellipsis">{name}</p>
        ):(
          <Button variant="link" className="font-medium p-0 h-6 justify-start" onClick={onOpen}>
            <p className="font-medium whitespace-nowrap overflow-hidden text-ellipsis">{name}</p>
          </Button>
        )}
        {(subtext !== undefined) && (
          <div className='h-4 flex flex-row space-x-4'>
            <p className={`text-xs text-primary/35 h-auto flex flex-col justify-center`}>{subtext.join(" | ")}</p>
            {(!finishedProcessing) && (
              <div className="h-4 bg-accent flex flex-row space-x-1 text-nowrap px-2 rounded-full">
                <div className='h-auto flex flex-col justify-center'>
                  <div style={{
                    animation: 'spin 1.5s linear infinite'
                  }}>
                    <LucideLoader2 className="w-3 h-3 text-primary" />
                  </div>
                </div>
                <p className='text-xs h-auto flex flex-col justify-center'>Processing</p>
              </div>
            )}
          </div>
        )}
      </div>
      {progress !== undefined && (
        <div className="w-[30%] px-2 pr-4">  
          <Progress value={progress} className='h-2 mb-2 w-auto' />
        </div>
      )}
      {(onDelete !== undefined) && (
        <Button size="icon" variant="ghost" onClick={onDelete}>
          <TrashIcon className="w-4 h-4" />
        </Button>
      )}
    </div>
  );
}

type uploading_file_type = {
  title: string,
  progress: number,
}

export default function CollectionPage() {

  const router = useRouter();
  const params = useParams<{ slug?: string[] | string }>()!;
  const slug = useMemo(() => {
    if (!params.slug) return [];
    return Array.isArray(params.slug) ? params.slug : [params.slug];
  }, [params.slug]);
  const collection_mode_immediate = (["create", "edit", "view"].indexOf(slug[0]) > -1) ? slug[0] as collection_mode_type : undefined
  const [CollectionMode, setCollectionMode] = useState<collection_mode_type>(collection_mode_immediate);
  const [collectionTitle, setCollectionTitle] = useState<string>("");
  const [collectionDescription, setCollectionDescription] = useState<string>("");
  const [collectionDocuments, setCollectionDocuments] = useState<fetch_collection_document_type[]>([]);
  const [collectionIsPublic, setCollectionIsPublic] = useState<boolean>(false);
  const [collectionOwner, setCollectionOwner] = useState<string>("personal");
  const [publishStarted, setPublishStarted] = useState<boolean>(false);
  const [uploadingFiles, setUploadingFiles] = useState<uploading_file_type[]>([]);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[] | null>(null);


  const {
    userData,
    refreshCollectionGroups,
  } = useContextAction();

  const fetchCollectionCallback = useCallback((only_documents : boolean) => {
    if (!slug[1]) return;

    fetchCollection({
      auth: userData?.auth as string,
      collection_id: slug[1],
      onFinish: (data) => {
        if (data !== undefined) {
          
          if (!only_documents) {
            setCollectionTitle(data.title);
            setCollectionDescription(data.description);
            setCollectionIsPublic(data.public);
          }

          fetchCollectionDocuments({
            auth: userData?.auth as string,
            collection_id: slug[1],
            limit: 100,
            offset: 0,
            onFinish: (data) => {
              if (data !== undefined) {
                setCollectionDocuments(data);
              }
            }
          })
        }
      }
    })
  }, [slug, userData?.auth]);

  useEffect(() => { // Keep refreshing collection documents every 5s if they are still processing
    let documents_processing = false;
    collectionDocuments.forEach(doc => {
      if (!doc.finished_processing) {
        documents_processing = true;
      }
    });
    if (documents_processing) {
      setTimeout(() => {
        fetchCollectionCallback(true);
      }, 5000)
    }
  }, [collectionDocuments, fetchCollectionCallback]);

  useEffect(() => {
    if ( userData?.auth !== undefined) {
      if (CollectionMode === "edit" || CollectionMode === "view") {
        // setCollectionMode(collection_mode_immediate)
        fetchCollectionCallback(false);
      }
    }
  }, [CollectionMode, fetchCollectionCallback, userData?.auth])

  const onPublish = () => {
    const create_args = {
      auth: userData?.auth as string,
      title: collectionTitle,
      description: collectionDescription,
      public: collectionIsPublic,
    }

    if (CollectionMode === "create") {
      createCollection({
        ...create_args, 
        ...(collectionOwner === "personal") ?
                                            {} :
                                            {organization_id: collectionOwner},
        onFinish: (result : false | {hash_id : string}) => {
          if (result !== false) {
            if (pendingUploadFiles !== null) {
              start_document_uploads(result.hash_id);
            } else {
              router.push(`/collection/edit/${result.hash_id}`);
            }
          }
        }
      });
    } else {
      modifyCollection({
        ...create_args,
        collection_id: slug[1],
        onFinish: (result) => {
          if (result !== false && pendingUploadFiles !== null) {
            start_document_uploads(slug[1]);
          }
          refreshCollectionGroups();
        }
      })
      
    }
  }

  const start_document_uploads = async (collection_hash_id : string) => {
    if (pendingUploadFiles === null) return;

    const url_2 = craftUrl("/upload/", {
      "auth": userData?.auth as string,
      "collection_hash_id": collection_hash_id
    });
  
    setPublishStarted(true);
    
    setUploadingFiles(pendingUploadFiles.map((file) => {
      return {
        title: file.name,
        progress: 0
      }
    }));

    const totalCount = pendingUploadFiles.length;

    for (let i = 0; i < pendingUploadFiles.length; i++) {
      const file = pendingUploadFiles[i];
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await axios.post(url_2.toString(), formData, {
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded / (progressEvent.total || 1)) * 100);
            console.log(`File ${file.name} upload progress: ${progress}%`, progress);
            setUploadingFiles(files => [{...files[0], progress: progress}, ...files.slice(1)]);
            // setUploadingFiles(files => [...files.slice(0, i), {...files[i], progress: progress}, ...files.slice(i+1)]);
          },
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        const response_data = response.data as {success: false} | {success : true, result: fetch_collection_document_type};

        if (response_data.success) {
          setUploadingFiles(files => files.slice(1));
          setCollectionDocuments((docs) => [response_data.result,...docs]);
          if (i === totalCount - 1) {
            onFinishedUploads(collection_hash_id);
          }
        }
        setPendingUploadFiles((files) => files?.filter((_, j) => j !== 0) || null)
        console.log("Pending Uploading Files", pendingUploadFiles.length);

      } catch (error) {
        console.error(`Failed to upload file ${file.name}:`, error);
      }
    }
  };

  const onFinishedUploads = (collection_hash_id : string) => {
    setPublishStarted(false);
    setPendingUploadFiles(null);
    refreshCollectionGroups();
    if (CollectionMode === "create")
      router.push(`/collection/edit/${collection_hash_id}`);
  }

  return (
    <ScrollArea className="w-full h-[calc(100vh)]">
      <div className="p-4 flex flex-col items-center w-full min-h-[calc(100vh)]">
        <h1 className="text-3xl font-bold tracking-tight mb-4 text-center">
          {(CollectionMode === "create")?"Create a Document Collection":(CollectionMode === "edit")?"Edit Document Collection":(CollectionMode === "view")?"View Document Collection":"Bad URL!"}
        </h1>
        {(slug[1]) && (
          <span className="text-xs text-primary/35 flex flex-row space-x-4 pb-10">
            <p className="text-sm text-nowrap overflow-hidden h-7 border-2 p-1 px-3 rounded-lg flex flex-col justify-center" >
              {slug[1]}
            </p>
            <Button type="submit" className="p-0 m-0 h-7" variant={"transparent"} onClick={() => {
              handleCopy(slug[1] || "")
            }}>
              <Copy className="w-4 h-4 text-primary"/>
            </Button>
          </span>
        )}
        <div className="grid px-4 md:grid-cols-2 md:gap-8 w-[80%] md:w-full flex-grow max-w-5xl pb-8">
          <div className="gap-2 grid-cols-2 flex flex-col space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="visibility">Visibility</Label>
                <Select value={(collectionIsPublic?"public":"private")} onValueChange={(value : string) => {
                  setCollectionIsPublic(value === "public");
                }} disabled={(CollectionMode !== "create" && CollectionMode !== "edit")}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select visibility" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="public">Public</SelectItem>
                    <SelectItem value="private">Private</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="owner">Owner</Label>
                <Select value={collectionOwner} onValueChange={(value : string) => {
                  setCollectionOwner(value);
                }} disabled={(CollectionMode !== "create")}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select owner" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="personal">Personal</SelectItem>
                    {(userData !== undefined) && userData.memberships.map((membership, index) => (
                      <SelectItem key={index} value={membership.organization_id}>{membership.organization_name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid w-full items-center gap-1.5">
              <Label htmlFor="title">Title</Label>
              
              <Input
                onChange={(event) => {
                  setCollectionTitle(event.target.value);
                }}
                value={collectionTitle}
                id="title" 
                placeholder={(slug[0] === "create")?"Enter title":""}
                disabled={(slug[0] === "view")}
              />
            </div>
            <div className="w-full items-center gap-1.5 flex-grow flex flex-col">
              <Label htmlFor="description" className='w-full text-left'>Description</Label>
              <Textarea
                className='flex-grow resize-none'
                id="description"
                value={collectionDescription}
                disabled={(slug[0] === "view")}
                onChange={(event) => {
                  setCollectionDescription(event.target.value);
                }}
                placeholder="Enter description"
              />
            </div>
          </div>
          <div className="gap-4 pt-4 md:pt-0 flex flex-col">
            {(CollectionMode === "create" || CollectionMode === "edit") && (
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="document">Upload Document</Label>
                <Input id="document" className='items-center text-center pb-0' type="file" multiple onChange={(event) =>{
                  if (event.target.files !== null) {
                    setPendingUploadFiles(Array.from(event.target.files));
                  }
                }}/>
              </div>
            )}
            <div className="h-72 w-full rounded-md border border-input flex-grow">
              <ScrollArea className="p-4 pt-2 text-sm h-full">
                <div className='w-[inherit]'>
                {(pendingUploadFiles !== null && !publishStarted) && pendingUploadFiles.map((file, index) => (
                  <FileDisplayType
                    key={index}
                    finishedProcessing={true}
                    name={file.name}
                    subtext={[file_size_as_string(file.size)]}
                    onDelete={() => {
                      setPendingUploadFiles(pendingUploadFiles.filter((_, i) => i !== index));
                    }} 
                  />
                ))}

                {uploadingFiles.map((file, index) => (
                  <FileDisplayType
                    key={index}
                    finishedProcessing={true}
                    name={file.title}
                    progress={file.progress} 
                  />
                ))}
                {collectionDocuments.map((doc, index) => (
                  <FileDisplayType
                    key={index} 
                    finishedProcessing={doc.finished_processing}
                    name={doc.title}
                    onOpen={() => {
                      openDocument({
                        auth: userData?.auth as string,
                        document_id: doc.hash_id
                      })
                    }}
                    subtext={[doc.size]} 
                    // progress={doc.progress} 
                    onDelete={(CollectionMode === "create" || CollectionMode === "edit")?(() => {
                      deleteDocument({
                        auth: userData?.auth as string,
                        document_id: doc.hash_id,
                        onFinish: (success : boolean) => {
                          if (success) {
                            const newDocs = collectionDocuments.filter((_, i) => i !== index);

                            setCollectionDocuments(newDocs);
                          }
                        }
                      })
                    }):undefined}
                  />
                ))}
                </div>
              </ScrollArea>
            </div>
            {(CollectionMode === "create" || CollectionMode === "edit") && (
              <Button disabled={publishStarted} className="w-full h-10" type="submit" onClick={onPublish}>
                Publish
              </Button>
            )}
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}

function TrashIcon(props : SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  )
}
