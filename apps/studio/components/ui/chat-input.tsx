import { useRef, useState, useCallback, useEffect, ChangeEvent } from "react";
import useAutosizeTextArea from "@/hooks/use-autosize-text-area";
import { ScrollArea, ScrollBar } from "./scroll-area";
import { cn } from "@/lib/utils";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"
import { Button } from "./button";
import { Paperclip, Send, Trash } from "lucide-react";
import { ClassValue } from "clsx";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"

export function FilePreview({
  className = "",
  fileName, 
  onDelete
}:{
  className?: string,
  fileName: string,
  onDelete: () => void
}) {
  const [hover, setHover] = useState(false);

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <div 
          className={className}
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
        >
          <div className="flex flex-row space-x-1 p-1 w-[inherit] max-w-[inherit]">
            {/* <div className="overflow-hidden whitespace-nowrap overflow-ellipsis"> */}
              <p className="overflow-hidden whitespace-nowrap overflow-ellipsis text-xs">{fileName}</p>
            {/* </div> */}
            {hover && (
              <div className="flex flex-row space-x-1">
                <button
                  // variant={"ghost"}
                  className="h-4 w-4 p-0 m-0"
                  onClick={onDelete}
                  >
                  <Trash className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>
          
        </div>
      </HoverCardTrigger>
      <HoverCardContent>
        <p>{fileName}</p>
      </HoverCardContent>
    </HoverCard>
  );
}

export default function ChatInput({
  className = "",
  upload = true,
  multiple = true,
  onSubmission = undefined,
  style = {},
}:{
  className?: ClassValue,
  upload?: boolean,
  multiple?: boolean,
  onSubmission?: (value : string, files: File[]) => void,
  style?: React.CSSProperties
}) {
  const [value, setValue] = useState("");
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [preventUpdate, setPreventUpdate] = useState(false);
  const [files, setFiles] = useState<File[]>([]);

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
      onSubmission?handleSubmission():null;
      setPreventUpdate(true);
      setValue("");
    }
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (files_in : FileList) => {
    if (files_in !== null && files_in.length > 0) {
      if (multiple) {
        setFiles([...files, ...Array.from(files_in)]);
      } else {
        setFiles([Array.from(files_in)[0]]);
      }
    }
  };

  const handleSubmission = useCallback(() => {
    if (onSubmission !== undefined) onSubmission(value, files);
    setValue("");
    setFiles([]);
  }, [value, files, onSubmission]);


  return (

    <div id="chat_input" className={cn(
      "max-h-[200px] flex w-full rounded-md bg-input text-sm ring-offset-background placeholder:text-muted-foreground",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed",
      className
    )} 
    style={style}
    onDrop={(e) => {
      e.preventDefault();
      handleFileChange(e.dataTransfer.files);
    }}
    onDragOver={(e) => {
      e.preventDefault();
    }}>
      <div className={cn(
        "w-full relative overflow-hidden rounded-[inherit]", 
        "flex-grow pl-3 pr-2.5 py-1 gap-x-0 items-start flex flex-row justify-between"
      )}>
        <ScrollArea className="h-full flex-grow rounded-[inherit] pb-0 flex flex-col justify-center">
          <div className="w-auto min-h-[30px] flex flex-col justify-center pr-2">
          {/* <div className="w-auto"> */}
            {(files.length > 0) && (
              <div className="w-auto flex flex-wrap gap-x-2 gap-y-1 pb-1">
                {files.map((file, index) => (
                  <FilePreview
                    key={index}
                    className="w-auto rounded-full bg-background px-2 text-sm text-primary max-w-[100px]" 
                    fileName={file.name}
                    onDelete={() => {
                      setFiles(files.filter((_, i) => i !== index));
                    }}
                  />
                ))}
              </div>
            )}
            <textarea
              className={"border-none pt-[calc(6px)] p-0 m-0 border-transparent flex-grow overflow-hidden outline-none resize-none bg-primary/0 border-0 ring-0 focus-visible:border-0 focus-visible:ring-0 ring-offset-0"}
              // className="items-center pt-[6px]"
              id="review-text"
              onChange={handleChange}
              placeholder="Message"
              ref={textAreaRef}
              rows={1}
              value={value}
              spellCheck={false}
              onKeyDown={onKeyDown}
            />
          {/* </div> */}
          </div>
        </ScrollArea>
        <div className="h-full flex flex-col justify-center">
          <div className="flex grid-flow-col space-x-1">
            {upload && (
              <>
                <input
                  type="file"
                  multiple={multiple}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={(e) => {if (e.target.files) handleFileChange(e.target.files)}}
                />
                <Button 
                  className="rounded-full p-0 m-0 h-8 w-8"
                  variant="ghost" 
                  type="submit" 
                  size="icon"
                  onClick={handleButtonClick}
                >
                  <Paperclip className="h-4 w-4 text-primary" />
                </Button>
              </>
            )}
            <Button 
              className="rounded-full p-0 pt-[1px] pr-[1px] m-0 h-8 w-8"
              variant="secondary"
              type="submit" 
              size="icon" 
              disabled={(value.length < 1 && files.length < 1)} 
              onClick={handleSubmission}
            >
              <Send className="h-4 w-4 text-primary" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
