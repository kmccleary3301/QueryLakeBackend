import "@/styles/globals.css"
import { Metadata } from "next"

import { siteConfig } from "@/config/site"
import { fontSans } from "@/lib/fonts"
import { cn } from "@/lib/utils"
import { Analytics } from "@/components/inherited/analytics"
import { ThemeProvider } from "@/components/inherited/providers"
import { StateThemeProvider, ThemeProviderWrapper } from "./theme-provider";
import { TailwindIndicator } from "@/components/inherited/tailwind-indicator"
import { ThemeSwitcher } from "@/components/inherited/theme-switcher"
import { Toaster as DefaultToaster } from "@/components/ui/toaster"
import { Toaster as NewYorkSonner } from "@/registry/new-york/ui/sonner"
import { Toaster as NewYorkToaster } from "@/registry/new-york/ui/toaster"
import { ContextProvider } from "./context-provider"
import RouteShell from "./route-shell";
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'

// import { useRouter } from "next/navigation"
// import { AppProps } from "next/app"
// import { AppProps, useRouter } from 'next/app'
// import { AnimatePresence } from 'framer-motion'

export const metadata: Metadata = {
  title: {
    default: siteConfig.name,
    template: `%s - ${siteConfig.name}`,
  },
  metadataBase: new URL(siteConfig.url),
  description: siteConfig.description,
  keywords: [
    "Next.js",
    "React",
    "Tailwind CSS",
    "Server Components",
    "Radix UI",
    "QueryLake",
    "LSU",
    "AI"
  ],
  authors: [
    {
      name: "Kyle McCleary",
      url: "https://github.com/kmccleary3301",
    },
  ],
  creator: "kmccleary3301",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
    images: [
      {
        url: siteConfig.ogImage,
        width: 1200,
        height: 630,
        alt: siteConfig.name,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteConfig.name,
    description: siteConfig.description,
    images: [siteConfig.ogImage],
    creator: "@shadcn",
  },
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon-16x16.png",
    apple: "/apple-touch-icon.png",
  },
  manifest: `${siteConfig.url}/site.webmanifest`,
}

export const viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "black" },
  ],
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  // const router = useRouter();
  // const pageKey = router.as;

  return (
    <>
      <html lang="en" suppressHydrationWarning className={`${GeistSans.variable} ${GeistMono.variable}`}>
        <head />
        <body
          className={cn(
            "min-h-screen bg-background antialiased font-geist-sans font-normal",
            GeistSans.className,
          )}
          style={{ scrollBehavior: "smooth"}}
        >
          {/* <ThemeWrapper> */}
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            // disableTransitionOnChange
          >
            <div vaul-drawer-wrapper="">
              <ContextProvider 
                userData={undefined} 
                selectedCollections={new Map()}
                toolchainSessions={new Map()}
              >
              <StateThemeProvider>
              <ThemeProviderWrapper>
              <RouteShell>
                {children}
              </RouteShell>
              <NewYorkSonner />
              </ThemeProviderWrapper>
              </StateThemeProvider>
              </ContextProvider>
            </div>
            <TailwindIndicator />
            <ThemeSwitcher />
            <Analytics />
            
            
          </ThemeProvider>
          {/* </ThemeWrapper> */}
        </body>
      </html>
    </>
  )
}
