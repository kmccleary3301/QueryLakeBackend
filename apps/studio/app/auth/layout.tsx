import { Metadata } from "next"
import { WavyBackground } from "./components/wavy-background";
import { QueryLakeDisplay } from "./components/display-text";
import React from "react";

export const metadata: Metadata = {
  title: "Log-In/Register",
  description: "Log in or register for QueryLake",
}

export default function CoverTestPage({ children }:{ children: React.ReactNode }) {
  return (
    <>
      <div style={{
        "--theme-one": "0 72.2% 50.6%",
        "--background": "0 0% 3.9%",
        "--background-sidebar": "0 0% 0%",
        "--foreground": "0 0% 98%",
        "--card": "0 0% 3.9%",
        "--card-foreground": "0 0% 98%",
        "--popover": "0 0% 3.9%",
        "--popover-foreground": "0 0% 98%",
        "--primary": "0 0% 98%",
        "--primary-foreground": "0 85.7% 97.3%",
        "--secondary": "0 0% 14.9%",
        "--secondary-foreground": "0 0% 98%",
        "--muted": "0 0% 14.9%",
        "--muted-foreground": "0 0% 63.9%",
        "--accent": "0 0% 14.9%",
        "--accent-foreground": "0 0% 98%",
        "--destructive": "0 62.8% 30.6%",
        "--destructive-foreground": "0 0% 98%",
        "--border": "0 0% 14.9%",
        "--input": "0 0% 14.9%"
      } as React.CSSProperties}>
        <WavyBackground 
          className="w-full mx-0 h-full"
          containerClassName="transform-gpu w-full h-[calc(100vh)]"
          canvasClassName="h-[calc(100vh)] w-[calc(100vw)] blur-[3px]"
          blur={3}
          waveWidth={4} 
          waveCount={20} 
          waveAmplitude={0.34}
          wavePinchEnd={0}
          wavePinchMiddle={0.064}
          speed={2.5}
          backgroundFill="primary"
        >
          <div className="w-full h-full">
            <div/>
            <QueryLakeDisplay>
              {children}
            </QueryLakeDisplay>
          </div>
        </WavyBackground>
      </div>
    </>
  )
}
