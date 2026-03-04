"use client";

import { componentMetaDataType, displayMapping } from "@/types/toolchain-interface";

export const METADATA : componentMetaDataType = {
    label: "Test Component",
    type: "Display",
    category: "Text Display",
    description: "This is a test component."
};

export default function NewTestComponent({
	configuration,
    demo = false
}:{
	configuration: displayMapping,
    demo?: boolean
}) {

    return (
        <div>
            <span className="text-red-500 text-4xl">TEST COMPONENT</span>
        </div>
    )
}