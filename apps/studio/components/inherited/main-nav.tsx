"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"

import { siteConfig } from "@/config/site"
import { cn } from "@/lib/utils"
import { Icons } from "@/components/inherited/icons"
import { Badge } from "@/registry/new-york/ui/badge"

export function MainNav() {
  const pathname = usePathname()

  return (
    <div className="mr-4 hidden md:flex">
      <Link href="/" className="mr-6 flex items-center space-x-2">
        <Icons.logo className="h-6 w-6" />
        <span className="hidden font-bold sm:inline-block">
          {siteConfig.name}
        </span>
      </Link>
      <nav className="flex items-center gap-6 text-sm">
        <Link
          href="/collection/12038/19192?a=312"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/collection")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Collection
        </Link>
        <Link
          href="/themes"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/themes")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Themes
        </Link>
        <Link
          href="/temes"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/temes")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Temes 782
        </Link>
        <Link
          href="/nodes"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/nodes")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Node Test
        </Link>
        <Link
          href="/aceternity"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/aceternity")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Aceternity
        </Link>
        <Link
          href="/examples"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/examples")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Examples
        </Link>
        <Link
          href="/auth"
          className={cn(
            "transition-colors hover:text-foreground/80",
            pathname?.startsWith("/auth")
              ? "text-foreground"
              : "text-foreground/60"
          )}
        >
          Auth
        </Link>
        <Link
          href={siteConfig.links.github}
          className={cn(
            "hidden text-foreground/60 transition-colors hover:text-foreground/80 lg:block"
          )}
        >
          GitHub
        </Link>
      </nav>
    </div>
  )
}
