"use client";

import { ScrollArea, ScrollAreaHorizontal } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useContextAction } from "@/app/context-provider";
import { useEffect, useState } from "react";
import { user_organization_membership, QueryLakeCreateOrganization, QueryLakeFetchUsersMemberships, organization_memberships, QueryLakeFetchOrganizationsMemberships, QueryLakeInviteUserToOrg, memberRoleLower } from "@/hooks/querylakeAPI";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

interface OrgPageProps {
  params: {
    slug: string[],
  },
  searchParams: object
}

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
import { toast } from 'sonner';
import { ComboBox, ComboBoxScrollPreview } from "@/components/ui/combo-box";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ArrowUpRight } from "lucide-react";
import { useParams } from "next/navigation";

export type memberRole = "Owner" | "Admin" | "Member" | "Viewer";

export default function InviteUserToOrgSheet({
  children,
  onSubmit,
}:{
  children: React.ReactNode,
  onSubmit: (form : {
    name: string,
    role: memberRole
  }) => void
}) {
  const [name, setName] = useState("");
  const [role, setRole] = useState<memberRole>("Viewer");


  return (
    <Sheet>
      <SheetTrigger asChild>
        {children}
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Invite User</SheetTitle>
          <SheetDescription>
            Invite a user to join this organization.
          </SheetDescription>
        </SheetHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right">
              User
            </Label>
            <Input id="name" value={name} onChange={
              (e) => setName(e.target.value)
            } placeholder="Username to invite" className="col-span-3" />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="role" className="text-right">
              Role
            </Label>
            <ComboBoxScrollPreview
              values={[
                {
                  value: "Owner",
                  label: "Owner",
                  preview: "Can delete organization and manage all members."
                },
                {
                  value: "Admin",
                  label: "Admin",
                  preview: "Can manage all members and document collections."
                },
                {
                  value: "Member",
                  label: "Member",
                  preview: "Can view and edit document collections."
                },
                {
                  value: "Viewer",
                  label: "Viewer",
                  preview: "Can view organization data and read documents."
                },
              ]}
              onChange={(value) => {
                setRole(value as memberRole);
              }}
              value={role}
            />
          </div>
        </div>
        <SheetFooter >
          <SheetClose asChild>
            <Button type="submit" variant={"secondary"} disabled={(name==="")} onClick={() => {onSubmit({name: name, role: role})}}>
              Send Invite
            </Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
