
import { 
  displayComponents as displayComponentsGenerated,
  inputComponents as inputComponentsGenerated,
  DISPLAY_COMPONENTS as DISPLAY_COMPONENTS_GENERATED,
  INPUT_COMPONENTS as INPUT_COMPONENTS_GENERATED,
  INPUT_COMPONENT_FIELDS as INPUT_COMPONENT_FIELDS_GENERATED,
} from "@/public/cache/toolchains/toolchain-component-types"


export type displayComponents = displayComponentsGenerated;
export type inputComponents = inputComponentsGenerated;

export const DISPLAY_COMPONENTS : displayComponents[] = DISPLAY_COMPONENTS_GENERATED;
export const INPUT_COMPONENTS : inputComponents[] = INPUT_COMPONENTS_GENERATED;

export type configEntryFieldType = {
  name: string,
  type: "boolean",
  default: boolean
} | {
  name: string,
  type: "string",
  default: string
} | {
  name: string,
  type: "number",
  default: number
} | {
  name: string,
  type: "long_string",
  default: string
}

export type inputComponentConfig = {
	hooks: string[],
	config?: configEntryFieldType[],
	tailwind?: string,
}


/*  This is for the toolchain interface designer.
 *  The fields config immediately below is for each input component.
 *  Note that the hook `selected_collections` is a special hook 
 *  that is used to send the selected collections to a toolchain event.
 */

export type componentMetaDataType = {
  label: string,
  type: "Display" | "Input",
  category: string,
  description?: string,
  config?: inputComponentConfig
}

// export const INPUT_COMPONENT_FIELDS : {[key in inputComponents]: inputComponentConfig} = {
// 	"file_upload": {
// 		"hooks": [
//       "on_upload",
//       "selected_collections"
//     ],
// 		"config": [
// 			{
// 				"name": "multiple",
// 				"type": "boolean",
//         "default": false
// 			}
// 		],
// 	},
// 	"chat_input": {
// 		"hooks": [
//       "on_upload", 
//       "on_submit",
//       "selected_collections",
//     ],
//     "config": [
//       {
// 				"name": "test_7_long_string",
// 				"type": "long_string",
//         "default": "6ix"
// 			}
// 		],
// 	},
//   "switch": {
// 		"hooks": [
//       "value_map",
//     ],
//     "config": [
//       {
// 				"name": "Label",
// 				"type": "string",
//         "default": ""
// 			}
// 		],
// 	},
// }
export const INPUT_COMPONENT_FIELDS : {[key in inputComponents]: inputComponentConfig} = INPUT_COMPONENT_FIELDS_GENERATED;

export type displayMapping = {
  display_route: (string | number)[],
  display_as: displayComponents;
}

export type inputEvent = {
  hook: string,
  target_event: string,
  fire_index: number,
  // target_route: (string | number)[],
  store: boolean,
  target_route: string,
}

export type configEntry = {
  name: string,
  value: string | number | boolean
}

export type configEntriesMap = Map<string, configEntry>;

// export type inputMappingProto = {
//   hooks: inputEvent[],
//   config: configEntry[],
//   tailwind: string,
// }

// export type fileUploadMapping = inputMappingProto & {
//   display_as: "file_upload"
// }

// export type chatInputMapping = inputMappingProto & {
//   display_as: "chat_input"
// }



// export type inputMapping = fileUploadMapping | chatInputMapping;

export type inputMapping = {
  display_as: inputComponents,
  hooks: inputEvent[],
  config: configEntry[],
  tailwind: string,
}

export type contentMapping = displayMapping | inputMapping;

export type alignType = "left" | "center" | "right" | "justify";

export type contentDiv = {
  type: "div",
  align: alignType,
  tailwind: string,
  mappings: (contentMapping | contentDiv)[]
}

export type contentSection = {
  split: "none",
  size: number,
  align: alignType,
  tailwind: string,
  mappings: (contentMapping | contentDiv)[],
  header?: headerSection,
  footer?: headerSection
}

export type divisionSection = {
  split: "horizontal" | "vertical",
  size: number,
  sections: displaySection[],
  header?: headerSection,
  footer?: headerSection
}


export type headerSection = {
  align: alignType,
  tailwind: string,
  mappings: (contentMapping | contentDiv)[]
}

export type displaySection = contentSection | divisionSection
