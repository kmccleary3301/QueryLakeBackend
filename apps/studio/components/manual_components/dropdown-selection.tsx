import { useState, useEffect } from "react"
import { CaretSortIcon, CheckIcon } from "@radix-ui/react-icons"
 
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
// import { use } from "marked"

export type formValueType = string | string[];

export type formEntryType = {
	label : string,
	value : formValueType
}

type DropDownSelectionProps = {
  values: formEntryType[],
  defaultValue: formEntryType,
  setSelection: (value : formEntryType) => void,
  selection: formEntryType,
  width?: number,
  display_values_name?: string
};

export function DropDownSelection(props : DropDownSelectionProps) {
  const [open, setOpen] = useState(false);
  // const [value, setValue] = useState<formValueType>(props.defaultValue.value);

  const displayValueName = (props.display_values_name)?(" "+props.display_values_name):"";

  useEffect(() => {
    console.log("SELECTION CHANGED", props.selection);
  }, [props.selection]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="justify-between"
          style={{
            width: (props.width) ? props.width : 400
          }}
        >
          {props.selection
            ? props.values.find((framework) => framework.value === props.selection.value)?.label
            : "Select"+displayValueName+"..."}
          <CaretSortIcon className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0" style={{width: (props.width) ? props.width : 400}}>
        <Command>
          <CommandInput placeholder={"Search"+displayValueName+"..."} className="h-9" />
          <CommandEmpty>No framework found.</CommandEmpty>
          <CommandGroup>
            {props.values.map((framework : formEntryType, index : number) => (
              <CommandItem
                key={index}
                value={framework.label}
                onSelect={() => {
                  // setValue(framework.value === props.selection.value ? "" : framework.value);
                  console.log("SETTING VALUE", framework);
                  props.setSelection(framework);
                  setOpen(false)
                }}
              >
                {framework.label}
                <CheckIcon
                  className={cn(
                    "ml-auto h-4 w-4",
                    props.selection.value === framework.value ? "opacity-100" : "opacity-0"
                  )}
                />
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  )
}