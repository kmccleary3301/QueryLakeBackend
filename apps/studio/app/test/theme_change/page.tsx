"use client";
import { useCallback, useEffect, useRef, useState } from "react";
// import { ScrollArea } from "@radix-ui/react-scroll-area";
import { ScrollArea } from "@/components/ui/scroll-area";
// import { Textarea } from "@/registry/default/ui/textarea";
import Editor, { Monaco, OnMount } from '@monaco-editor/react';
import { editor } from "monaco-editor-core";
import { Button } from "@/components/ui/button";
import { get_all_language_backgrounds } from "@/lib/shiki";
import { hslStringToHsl, hslStringToRGBHex, hslToHslString, hslToRgb, rgbToHex, rgbToHsl } from "@/hooks/rgb-hsl-functions";
// import { getHighlighter } from 'shiki';
// import * as monaco from 'monaco-editor';

function ThemeController({
  children,
}:{
  children: React.ReactNode
}) {
  const [override, setOverride] = useState(false);

  const handleChangeColors = () => {
    // const currentClassList = document.body.classList;
    // console.log("CURRENT CLASS LIST:", currentClassList);
    setOverride((override) => !override);
  };

  return (
    <div
      style={(override)?({
        '--background': '0, 100%, 50%',
        '--foreground': '0, 100%, 50%',
        '--card': '0, 100%, 50%',
        '--card-foreground': '0, 100%, 50%',
        '--popover': '0, 100%, 50%',
        '--popover-foreground': '0, 100%, 50%',
        '--primary': '0, 100%, 50%',
        '--primary-foreground': '0, 100%, 50%',
        '--secondary': '0, 100%, 50%',
        '--secondary-foreground': '0, 100%, 50%',
        '--muted': '0, 100%, 50%',
        '--muted-foreground': '0, 100%, 50%',
        '--accent': '0, 100%, 50%',
        '--accent-foreground': '0, 100%, 50%',
        '--destructive': '0, 100%, 50%',
        '--destructive-foreground': '0, 100%, 50%',
        '--border': '0, 100%, 50%',
        '--input': '0, 100%, 50%',
      } as React.CSSProperties):{}} // Add type assertion to accept custom CSS properties
      className="custom-colors bg-primary-color text-secondary-color p-4"
    >
      <Button onClick={handleChangeColors}>Change Colors</Button>
      <Button onClick={() => {
        const hsl_test = "240 100% 50%";
        console.log("hsl_test:", hsl_test);
        const c_1 = hslStringToHsl(hsl_test) as number[];
        console.log("c_1:", c_1);
        const c_2 = hslToRgb(c_1);
        console.log("c_2:", c_2);
        const c_3 = rgbToHex(c_2);
        console.log("c_3:", c_3);
        const c_4 = rgbToHsl(c_2);
        console.log("c_4:", c_4);
        const c_5 = hslToHslString(c_4);
        console.log("c_5:", c_5);
        const c_6 = hslStringToRGBHex(hsl_test);
        console.log("c_6:", c_6);
      }}>Get all language backgrounds</Button>
      {children}
    </div>
  );
}

export default function ThemeTestPage() {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleEditorDidMount = (editor: editor.IStandaloneCodeEditor, monaco: Monaco) => {
    // here is the editor instance
    // you can store it in `useRef` for further usage
    editorRef.current = editor;
  }

  return (
    <div className="w-full h-[calc(100vh)] flex flex-row justify-center">
      <ScrollArea className="w-full">
        <div className="flex flex-row justify-center">
          <div className="w-[85vw] md:w-[70vw] lg:w-[45vw]">
            <ThemeController>
              <div className="flex flex-wrap justify-center gap-6 py-[100px]">
                <div className="w-[100px] h-[100px] bg-background border-8 border-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">background</p>
                </div>
                <div className="w-[100px] h-[100px] bg-foreground border-8 border-background">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-card border-8 border-card-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">card</p>
                </div>
                <div className="w-[100px] h-[100px] bg-card-foreground border-8 border-card">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">card foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-popover border-8 border-popover-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">popover</p>
                </div>
                <div className="w-[100px] h-[100px] bg-popover-foreground border-8 border-popover">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">popover foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-primary border-8 border-primary-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">primary</p>
                </div>
                <div className="w-[100px] h-[100px] bg-primary-foreground border-8 border-primary">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">primary foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-secondary border-8 border-secondary-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">secondary</p>
                </div>
                <div className="w-[100px] h-[100px] bg-secondary-foreground border-8 border-secondary">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">secondary-foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-muted border-8 border-muted-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">muted</p>
                </div>
                <div className="w-[100px] h-[100px] bg-muted-foreground border-8 border-muted">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">muted foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-accent border-8 border-accent-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">accent</p>
                </div>
                <div className="w-[100px] h-[100px] bg-accent-foreground border-8 border-accent">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">accent foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-destructive border-8 border-destructive-foreground">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">destructive</p>
                </div>
                <div className="w-[100px] h-[100px] bg-destructive-foreground border-8 border-destructive">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">destructive foreground</p>
                </div>
                <div className="w-[100px] h-[100px] bg-border border-8 border-background">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">border</p>
                </div>
                <div className="w-[100px] h-[100px] bg-input border-8 border-background">
                  <p className="w-auto h-auto flex flex-col justify-center bg-white text-black text-center text-xs">input</p>
                </div>
              </div>
            </ThemeController>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

// export CodeEditor;