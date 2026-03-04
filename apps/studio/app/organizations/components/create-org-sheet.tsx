"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";


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

export default function CreateOrgSheet({
  children,
  onSubmit,
}:{
  children: React.ReactNode,
  onSubmit: (form : {
    name: string
  }) => void
}) {
  const [name, setName] = useState("");


  return (
    <Sheet>
      <SheetTrigger asChild>
        {children}
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Create Organization</SheetTitle>
          <SheetDescription>
            Create a new QueryLake organization. 
          </SheetDescription>
        </SheetHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right">
              Name
            </Label>
            <Input id="name" value={name} onChange={
              (e) => setName(e.target.value)
            } placeholder="Set Organization Name" className="col-span-3" />
          </div>
        </div>
        <SheetFooter >
          <SheetClose asChild>
            <Button type="submit" variant={"secondary"} disabled={(name==="")} onClick={() => {onSubmit({name: name})}}>Create</Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}