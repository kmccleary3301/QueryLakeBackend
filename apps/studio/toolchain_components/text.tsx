"use client";
import { useToolchainContextAction } from "@/app/app/context-provider";
import { useContextAction } from "@/app/context-provider";
import { retrieveValueFromObj, toolchainStateType } from "@/hooks/toolchain-session";
import { componentMetaDataType, displayMapping } from "@/types/toolchain-interface";
import { useEffect, useState } from "react";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";
import { CHAT_RENDERING_STYLE } from "@/components/markdown/configs";

export const METADATA : componentMetaDataType = {
  label: "Text",
  type: "Display",
  category: "Text Display",
  description: "Displays text as-is, without rendering it as markdown or processing it in any way.",
};

const SAMPLE_TEXT = `
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?
`


export default function Text({
	configuration,
  demo = false
}:{
	configuration: displayMapping,
  demo?: boolean
}) {

  const { toolchainState, toolchainWebsocket } = useToolchainContextAction();
  const { userData } = useContextAction();

	const [currentValue, setCurrentValue] = useState<string>(
    demo ?
    SAMPLE_TEXT :
    retrieveValueFromObj(toolchainState, configuration.display_route) as string || ""
	);

  useEffect(() => {
		if (toolchainWebsocket?.current === undefined || demo) return;
    const newValue = retrieveValueFromObj(toolchainState, configuration.display_route) as string || "";
    // console.log("Chat newValue", JSON.parse(JSON.stringify(newValue)));
		setCurrentValue(newValue);
	}, [toolchainState]);

  return (
    <div className="max-w-full">
      <MarkdownRenderer
        disableRender
        config={CHAT_RENDERING_STYLE}
        input={currentValue} 
        finished={false}
      />
	  </div>
  );
}