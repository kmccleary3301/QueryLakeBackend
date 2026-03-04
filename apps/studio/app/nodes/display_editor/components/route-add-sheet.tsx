"use client";
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
	Sheet,
	SheetClose,
	SheetContent,
	SheetDescription,
	SheetFooter,
	SheetHeader,
	SheetTitle,
	SheetTrigger,
} from "@/components/ui/sheet"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { useRef, useState } from "react"

export default function RouteAddSheet({
	value = "",
	type,
	onChange,
	children
}:{
	value?: (string | number),
	type: "string" | "number",
	onChange: (value: string | number) => void,
	children: React.ReactNode
}) {

	const actingValue = useRef<string	| number>(value);
	const [entryType, setEntryType] = useState<"string" | "number">(type);

	return (
		<Sheet>
			<SheetTrigger asChild>
				{children}
			</SheetTrigger>
			<SheetContent className="space-y-2">
				<SheetHeader>
					<SheetTitle>Add Element</SheetTitle>
					<SheetDescription>
						Add an element to the route of the display object.
					</SheetDescription>
				</SheetHeader>
				<div className="grid gap-4 py-4">
					
					<div className="flex flex-col space-y-4 items-start">
						<div className="flex flex-row justify-between w-full">
							<div className="h-auto flex flex-col justify-center">
								<Label htmlFor="username" className="text-center text-lg">
									Value
								</Label>
							</div>
							<ToggleGroup
								type="single"
								onValueChange={(value : "" | "number") => {
									setEntryType((value === "number") ? "number" : "string");
								}}
								className='flex flex-row justify-between'
								value={(entryType === "number") ? "number" : ""}
							>
								<ToggleGroupItem value="number" aria-label="Align center" variant={"outline"}>
									Number
								</ToggleGroupItem>
							</ToggleGroup>
						</div>
						<Input 
							id="value" 
							defaultValue={value.toString()}
							placeholder="Enter a directory in the state"
							className="col-span-3" 
							spellCheck={false}
							onChange={(e) => {
								if (entryType === "string") {
									actingValue.current = e.target.value;
								} else {
									actingValue.current = parseInt(e.target.value);
								}
							}}
						/>
					</div>
				</div>
				<SheetFooter>
					<SheetClose asChild>
						<Button type="submit" className="w-full" onClick={() => onChange(actingValue.current)}>Add Element</Button>
					</SheetClose>
				</SheetFooter>
			</SheetContent>
		</Sheet>
	)
}