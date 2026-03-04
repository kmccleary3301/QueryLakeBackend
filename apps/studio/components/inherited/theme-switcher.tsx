"use client"

import * as React from "react"
import { useSelectedLayoutSegment } from "next/navigation"

import { useConfig } from "@/hooks/use-config"

export function ThemeSwitcher() {
  const [config] = useConfig()
  const segment = useSelectedLayoutSegment()

  React.useEffect(() => {
    console.log("CONFIG UPDATED:", config);
    
    document.body.classList.forEach((className) => {
      if (className.match(/^theme.*/)) {

        const currentClassList = document.body.classList;
        console.log("CURRENT CLASS LIST:", currentClassList);
        document.body.classList.remove(className)
      }
    })

    const theme = segment === "themes" ? config.theme : null
    if (theme) {
      return document.body.classList.add(`theme-${theme}`)
    }
  }, [segment, config])

  return null
}
