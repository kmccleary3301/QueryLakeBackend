import { inputComponentConfig } from "@/types/toolchain-interface";

export type displayComponents = "audio-recorder" | "chat" | "current-event-display" | "markdown" | "text";
export type inputComponents = "basf-intro-screen" | "chat-input" | "chat-with-rating" | "file-upload" | "switch";
export const DISPLAY_COMPONENTS : displayComponents[] = ["audio-recorder","chat","current-event-display","markdown","text"];
export const INPUT_COMPONENTS : inputComponents[] = ["basf-intro-screen","chat-input","chat-with-rating","file-upload","switch"];
export const INPUT_COMPONENT_FIELDS : {[key in inputComponents]: inputComponentConfig} = {
  "basf-intro-screen": {
    "hooks": [
      "on_upload",
      "on_submit",
      "selected_collections"
    ],
    "config": [
      {
        "name": "test_7_long_string",
        "type": "long_string",
        "default": "Hello, how are you?"
      }
    ]
  },
  "chat-input": {
    "hooks": [
      "on_upload",
      "on_submit",
      "selected_collections"
    ]
  },
  "chat-with-rating": {
    "hooks": [
      "on_rating"
    ],
    "config": [
      {
        "name": "Type",
        "type": "long_string",
        "default": "Boolean"
      },
      {
        "name": "Label",
        "type": "string",
        "default": "Rate this response"
      }
    ]
  },
  "file-upload": {
    "hooks": [
      "on_upload",
      "selected_collections"
    ],
    "config": [
      {
        "name": "multiple",
        "type": "boolean",
        "default": false
      }
    ]
  },
  "switch": {
    "hooks": [
      "value_map"
    ],
    "config": [
      {
        "name": "Label",
        "type": "string",
        "default": ""
      }
    ]
  }
};