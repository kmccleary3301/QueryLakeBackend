"use client";
import { retrieveValueFromObj, toolchainStateType } from "@/hooks/toolchain-session";
import { componentMetaDataType, displayMapping } from "@/types/toolchain-interface";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";
import { useToolchainContextAction } from "@/app/app/context-provider";
import { useContextAction } from "@/app/context-provider";
import { useEffect, useState } from "react";
import MARKDOWN_SAMPLE_TEXT from "@/components/markdown/demo-text";
import { CHAT_RENDERING_STYLE } from "@/components/markdown/configs";

export const METADATA : componentMetaDataType = {
	label: "Markdown",
  type: "Display",
	category: "Text Display",
	description: "Displays text as markdown.",
};

export default function Markdown({
	configuration,
  demo = false,
}:{
	configuration: displayMapping,
  demo?: boolean,
}) {

  const { toolchainState, toolchainWebsocket } = useToolchainContextAction();

	const [currentValue, setCurrentValue] = useState<string>(
    demo ?
    MARKDOWN_SAMPLE_TEXT :
    retrieveValueFromObj(toolchainState, configuration.display_route) as string || ""
	);

  useEffect(() => {
		if (toolchainWebsocket?.current === undefined || demo) return;
    const newValue = retrieveValueFromObj(toolchainState, configuration.display_route) as string || "";
    // console.log("Chat newValue", JSON.parse(JSON.stringify(newValue)));
		setCurrentValue(newValue);
	}, [toolchainState]);

  return (
    <div className="max-w-full p-0 -mt-1.5">
      <MarkdownRenderer
        // disableRender={(value.role === "user")}
        config={CHAT_RENDERING_STYLE}
        input={currentValue} 
        finished={false}
      />
	  </div>
  );
}