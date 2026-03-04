import { useRef, useState, useCallback, useEffect } from "react";
import useAutosizeTextArea from "@/hooks/use-autosize-text-area";
import { Textarea } from "@/components/ui/textarea";
// import "@/App.css";
import { ScrollArea, ScrollBar } from "./scroll-area";
// import { DragEventHandler } from "react";
import {useDropzone} from 'react-dropzone';
import { motion } from 'framer-motion';
import { cn } from "@/lib/utils";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"

// type FileDropEvent = React.DragEvent<HTMLDivElement>;

// type AdaptiveTextAreaProps = {
//   value: string,
//   setValue: React.Dispatch<React.SetStateAction<string>>,
//   onSubmission?: (value : string) => void,
//   onUpdateHeight?: (height : number) => void
//   onDrop?: (files : File[]) => void,
// }


export default function AdaptiveTextArea({
  value,
  setValue,
  onSubmission,
  style,
  ...props
}:{
  value: string,
  setValue: React.Dispatch<React.SetStateAction<string>>,
  onSubmission?: (value : string) => void,
  style?: React.CSSProperties
}) {
//   const [value, setValue] = useState("");
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [preventUpdate, setPreventUpdate] = useState(false);

  useAutosizeTextArea(textAreaRef.current, value);

  const handleChange = (evt: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = evt.target?.value;
    if (!preventUpdate) {
      setValue(val);
    }
    setPreventUpdate(false);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!event.shiftKey && event.key === "Enter" && value.length > 0) {
      onSubmission?onSubmission(value):null;
      setPreventUpdate(true);
      setValue("");
      // setTimeout(() => {}, 20);
    }
  };

  const [dragging, setDragging] = useState(false);

  const handleDragEnter = () => {
    // event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    // event.preventDefault();
    setDragging(false);
  };

  return (

    <div 
      className="max-h-[200px] flex w-full rounded-md border border-input bg-background text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <ScrollArea className={"flex-grow pl-3 py-2 gap-x-4 items-start"}>
        <div 
          className="w-auto flex flex-row pr-3"
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
        >
          <textarea
            className={"border-none border-transparent flex-grow overflow-hidden outline-none h-auto resize-none bg-primary/0 border-0 ring-0 focus-visible:border-0 focus-visible:ring-0 ring-offset-0"}
            id="review-text"
            onChange={handleChange}
            placeholder="Message"
            ref={textAreaRef}
            rows={1}
            value={value}
            spellCheck={false}
            onKeyDown={onKeyDown}
            
          />
        </div>
      </ScrollArea>
    </div>
  );
}
