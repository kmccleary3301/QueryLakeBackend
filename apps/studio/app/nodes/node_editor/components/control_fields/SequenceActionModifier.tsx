import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { HoverTextDiv } from "@/components/ui/hover-text-div";
import { Textarea } from "@/components/ui/textarea";
import { appendAction, backOut, createAction, deleteAction, operatorAction, sequenceAction, sequenceActionNonStatic, updateAction } from "@/types/toolchains";
import { ChevronLeft, Trash } from "lucide-react";

const SEQUENCE_ACTION_MAP: { [key: string]: string[] } = {
  "createAction":   ["bg-yellow-500", "C", "Create"   , "text-yellow-500", "border-yellow-500"],
  "updateAction":   ["bg-green-500",  "U", "Update"   , "text-green-500", "border-green-500"],
  "appendAction":   ["bg-green-500",  "A", "Append"   , "text-green-500", "border-green-500"],
  "deleteAction":   ["bg-red-500",    "D", "Delete"   , "text-red-500", "border-red-500"],
  "operatorAction": ["bg-purple-500", "+", "Operation", "text-purple-500", "border-purple-500"],
  "backOut":        ["bg-orange-500", "<", "Back Out" , "text-orange-500", "border-orange-500"],
  "string":         ["bg-blue-500",   ">", "String"   , "text-blue-500", "border-blue-500"],
  "number":         ["bg-blue-500",   ">", "Number"   , "text-blue-500", "border-blue-500"],
}

const SEQUENCE_ACTION_ICON_HINTS: { [key: string]: string } = {
  "createAction": "Create",
  "updateAction": "Update",
  "appendAction": "Append",
  "deleteAction": "Delete",
  "operatorAction": "Operation",
  "backOut": "Back Out",
  "string": "String",
  "number": "Number",
}


function SequenceActionStandard({
  type,
  children,
}:{
  type: string,
  children?: React.ReactNode,
}) {

  const sequence_values = SEQUENCE_ACTION_MAP[type] || ["bg-white", "?", "Unknown", "text-white", "border-white"];

  return (
    <div className="flex flex-row">
      <div>
      {/* <HoverTextDiv hint={sequence_values[2]}> */}

        <HoverCard>
          <HoverCardTrigger asChild>
            <div>
            <p className={cn(`rounded-full p-1 h-7 min-w-7 text-center font-bold text-black select-none`, sequence_values[0])}>
              {sequence_values[1]}
            </p>
            </div>
          </HoverCardTrigger>
          <HoverCardContent className="w-auto max-w-[200px] flex flex-row space-x-2">
            <Button variant={"ghost"} className="w-6 h-6 p-0 m-0 text-destructive hover:text-destructive/80 active:text-destructive/60">
              <Trash className="w-4 h-4"/>
            </Button>
            <p className="text-sm h-auto flex flex-col justify-center">
              {sequence_values[2]}
            </p>
          </HoverCardContent>
        </HoverCard>
        
      {/* </HoverTextDiv> */}
      </div>
      {/* <div>
        <ChevronLeft className={cn("h-7 -mr-4 pr-[2px] pb-[0px]", sequence_values[3])}/>
      </div> */}
      <div className="flex flex-row -mr-[1px] pl-2">
        <div className={cn("w-[15px] mb-5 border-t-2 border-r-2 rounded-tr-xl mr-[0.5px]", sequence_values[4])} style={{marginTop: "calc(0.875rem )"}}>

        {/* <div 
          className={cn(
            "h-3 w-3 border-b-2 border-l-2 flex flex-col justify-center -mr-3.5 m-[1px]", 
            sequence_values[3], 
            sequence_values[4]
          )}
          style={{
            marginTop: "calc(-100%)",
            // Rotate 45 degrees
            transform: "rotate(45deg)",
          }}
        /> */}
        </div>
        {/* <div className={cn("h-7 flex flex-col justify-center", sequence_values[3])}>
        </div> */}
      </div>
      <div className={cn("flex flex-col border-l-2 border-b-2 rounded-bl-xl -ml-[1px]", sequence_values[4])} style={{marginTop: "calc(0.875rem + 10px)"}}>
        <div className="pl-4 pb-4 -mt-3.5 min-h-[40px]">
        {children}
        </div>
      </div>
    </div>
  )
}



export default function SequenceActionModifier({
  data,
  deleteSelf,
  className = "",
}:{
  data : sequenceAction,
  deleteSelf?: () => void,
  className?: string,
}) {

  return(
    <>
      {((typeof data === "object") && 
        ((data as createAction).type === "createAction")) && (
          <SequenceActionStandard type="createAction">
            {((data as createAction).initialValue !== undefined) && (
              <div>
                <p>Initial Value</p>
                <Textarea className="resize-none" value={JSON.stringify((data as createAction).initialValue)}/>
              </div>
            )}
          </SequenceActionStandard>
      )}
      {((typeof data === "object") && 
        ((data as updateAction).type === "updateAction")) && (
          <SequenceActionStandard type="updateAction">
            {((data as createAction).initialValue !== undefined) && (
              <div/>
            )}
          </SequenceActionStandard>
      )}
      {((typeof data === "object") && 
        ((data as appendAction).type === "appendAction")) && (
          <SequenceActionStandard type="appendAction">
            {((data as appendAction).initialValue !== undefined) && (
              <div>
                <p>Initial Value</p>
                <Textarea value={JSON.stringify((data as appendAction).initialValue)}/>

              </div>
            )}
          </SequenceActionStandard>
      )}
      {((typeof data === "object") && 
        ((data as deleteAction).type === "deleteAction")) && (
        <SequenceActionStandard type="deleteAction">
          {((data as createAction).initialValue !== undefined) && (
            <div/>
          )}
        </SequenceActionStandard>
      )}
      {((typeof data === "object") && 
        ((data as operatorAction).type === "operatorAction")) && (
        <SequenceActionStandard type="operatorAction">
          {((data as createAction).initialValue !== undefined) && (
            <div/>
          )}
        </SequenceActionStandard>
      )}
      {((typeof data === "object") && 
        ((data as backOut).type === "backOut")) && (
          <SequenceActionStandard type="backOut">
            {((data as createAction).initialValue !== undefined) && (
              <div/>
            )}
          </SequenceActionStandard>
      )}
      {/* Index Route Retrieval */}



      {/* Const Values */}

      {((typeof data === "string") && 
        ((data as string))) && (
          <SequenceActionStandard type="string">
            <div/>
          </SequenceActionStandard>
      )}

      {((typeof data === "number") && 
        ((data as number))) && (
          <SequenceActionStandard type="number">
            
          </SequenceActionStandard>
      )}
    </>
  );
}