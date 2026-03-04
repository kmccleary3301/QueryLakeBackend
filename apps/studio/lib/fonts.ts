// import { JetBrains_Mono as FontMono, Inter as FontSans } from "next/font/google"
import { JetBrains_Mono as FontMono } from "next/font/google"
// import { GeistMono } from "geist/font/mono"
// import { GeistSans } from "geist/font/sans"
import localFont from 'next/font/local'
import { Inter } from "next/font/google";


// export const fontSans = FontSans({
//   subsets: ["latin"],
//   variable: "--font-sans",
// })




// export const fontSans = GeistSans

export const fontSans = localFont({
  src: [
    {
      path: '../assets/fonts/Geist/Geist-Regular.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../assets/fonts/Geist/Geist-Bold.woff2',
      weight: '700',
      // style: 'normal',
      style: 'bold',
    }
  ],
  variable: "--font-sans",
})

export const fontSoehne = localFont({
  src: [
    {
      path: '../assets/fonts/Soehne/soehne-buch.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../assets/fonts/Soehne/soehne-halbfett.woff2',
      weight: '700',
      style: 'bold',
    }
  ],
  variable: "--font-soehne",
})

export const fontMono = FontMono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export const fontConsolas = localFont({
  src: [
    {
      path: '../assets/fonts/Consolas/Consolas.ttf',
      weight: '300',
      style: 'normal',
    }
  ],
  variable: "--font-consolas",
})


export const fontInter = localFont({
  src: [
    {
      path: '../assets/fonts/Inter/medium.ttf',
      style: 'medium',
    },
    {
      path: '../assets/fonts/Inter/regular.ttf',
      style: 'normal',
    },
    {
      path: '../assets/fonts/Inter/semi-bold.ttf',
      style: 'semibold',
    }
  ],
  variable: "--font-inter",
});
