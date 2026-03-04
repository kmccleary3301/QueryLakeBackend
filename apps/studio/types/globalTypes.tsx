export type buttonCallback = {
  input_argument : string,
  type: "event_button",
  display_as: "button",
  feather_icon: string,
  button_text: string,
  return_file_response?: boolean
};

export type displayType = {
  input_argument : string,
  type: "<<STATE>>" | "event_offer",
  display_as: "chat" | "chat_window_files" | "center_table" | "chat_window_progress_bar" | "markdown"
} | buttonCallback | {
  input_argument : string, // In this case, it is a node id.
  type: "node_stream_temporary_output" | "node_stream_output",
  display_as: "chat_entry" | "markdown"
};

export type toolchainEntry = {
  name: string,
  id: string,
  category: string
  chat_window_settings: {
    display: displayType[],
    max_files: number,
    enable_rag: boolean,
    events_available: string[] //Needs to be a union type.
  }
};

export type toolchainCategory = {
  category: string,
  entries: toolchain_type[]
};



export type availableToolchainsResult = {
	default: toolchainEntry,
	toolchains: toolchainCategory[]
};

export type CodeSegmentExcerpt = {
  text: string,
  color: string,
};

export type CodeSegment = CodeSegmentExcerpt[];

export type ChatContentExcerpt = string | CodeSegment;

export type ChatContent = ChatContentExcerpt[];

export type sourceMetadata = {
  metadata: metadataDocumentRaw
};

export type metadataDocumentRaw = {
  "type": "pdf"
  "collection_type": "user" | "organization" | "global",
  "document_id": string,
  "document": string,
  "document_name": string,
  "location_link_chrome": string,
  "location_link_firefox": string,
  "page": number,
  "rerank_score"?: number
} | {
  "type": "web"
  "url": string, 
  "document": string,
  "document_name": string,
  "rerank_score"?: number
};

export type chatSourceMetadata = {
  icon?: string | ArrayBuffer | Blob,
  userData: userDataType,
  document : string,
  metadata: metadataDocumentRaw
}

//This type 
export type ChatEntry = {
  role: ("user" | "assistant" | "display"),
  // content_392098: ChatContent,
  content: string,
  // status?: ("generating_query" | "searching_google" | "typing"),
  sources?: sourceMetadata[],
  state?: "finished" | "searching_web" | "searching_vector_database" | "crafting_query" | "writing" | undefined
};

export type pageID = "ChatWindow" | "MarkdownTestPage" | "LoginPage" | "CreateCollectionPage" | "EditCollection" | "OrganizationManager" | "UserSettings";

export type selectedState = {
	selected: boolean,
	setSelected: React.Dispatch<React.SetStateAction<boolean>>,
};

export type collectionEntry = {
  title: string,
  hash_id: string,
  items: number,
  type: string // This only has specific possibilities. Update it to a union later.
};

export type collectionGroup = {
	title: string,
	toggleSelections?: selectedState[],
	selected?: selectedState,
	collections: collectionEntry[],
};

export type selectedCollectionsType = Map<string, boolean>;

export type sessionEntry = {
  time: number,
  title: string,
  hash_id: string
}

export type genericArrayType = Array<string | boolean | number | object>;
export type genericMapValueType = string | boolean | number | object | genericArrayType;
export type sessionStateType = Map<string, string | boolean | number | object | genericArrayType>;

export type membershipType = {
	organization_id: string,
	organization_name: string,
	role: "owner" | "admin" | "member" | "viewer",
	invite_still_open: boolean,
};

export type model_type = {
  title : string,
  id : string,
  model_description : string
};

export type external_model_provider = {
  provider_name : string,
  models : model_type[]
}

export type available_models = {
  default_model: model_type,
  local_models: model_type[],
  external_models?: external_model_provider[]
};

export type all_available_models = {
  [model_type : string]: available_models
}

export type toolchain_type = {
  title : string,
  id : string,
  category : string
}

export type toolchain_category = {
  category : string,
  entries : toolchain_type[]
}

export type toolchain_session = {
  title : string,
  toolchain: string,
  id : string,
  time : number
}

export type userDataType = {
	username: string,
	auth: string,
	memberships: membershipType[],
	is_admin: boolean,
	available_models: all_available_models,
	available_toolchains: toolchainCategory[],
  default_toolchain: toolchain_type,
  user_set_providers: string[],
  providers: string[],
};


export type setStateOrCallback<T> = React.Dispatch<React.SetStateAction<T>> | ((value: T) => void);


export type timeWindowType = {
  title: string,
  cutoff: number,
  entries: toolchain_session[]
};

export type QueryLakeApiKey = {
	id: string, 
	title: string, 
	key_preview: string,
	created: number,
  created_string?: string,
	last_used: number | null,
  last_used_string?: string,
};


export type APIFunctionArgument = {
  keyword: string,
  type_hint?: string,
  default_value?: any,
}

export type APIFunctionSpec = {
  endpoint: string,
  api_function_id: string,
  function_name: string,
  description: string,
  function_args: APIFunctionArgument[],
}
