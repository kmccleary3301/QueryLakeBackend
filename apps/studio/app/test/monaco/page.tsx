"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { ScrollArea } from "@radix-ui/react-scroll-area";
// import { Textarea } from "@/registry/default/ui/textarea";
import Editor, { Monaco, OnMount } from '@monaco-editor/react';
import { editor } from "monaco-editor-core";
// import { getHighlighter } from 'shiki';
// import * as monaco from 'monaco-editor';

const TEST_CODE = `
"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/registry/default/ui/button";
import ToolchainSession, { CallbackOrValue, ToolchainSessionMessage, toolchainStateType } from "@/hooks/toolchain-session";
import { useContextAction } from "@/app/context-provider";
import ChatBarInput from "@/components/manual_components/chat-input-bar";
import FileDropzone from "@/registry/default/ui/file-dropzone";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { Textarea } from "@/registry/default/ui/textarea";
// import initializeEditor from "./components/code_editor";
// import { getHighlighter } from 'shiki';
// import { shikiToMonaco } from '@shikijs/monaco';
// import * as monaco from 'monaco-editor-core';
import Editor, { Monaco, OnMount } from '@monaco-editor/react';
import { editor } from "monaco-editor-core";
import { getHighlighter } from 'shiki';
// import { shikiToMonaco } from '@shikijs/monaco';
// import * as monaco from 'monaco-editor-core';

// const CodeEditor = () => {
//   const containerRef = useRef(null);

//   useEffect(() => {
//     const initializeEditor = async () => {
//       const highlighter = await getHighlighter({
//         themes: ['vitesse-dark', 'vitesse-light'],
//         langs: ['javascript', 'typescript', 'vue'],
//       });

//       monaco.languages.register({ id: 'vue' });
//       monaco.languages.register({ id: 'typescript' });
//       monaco.languages.register({ id: 'javascript' });

//       shikiToMonaco(highlighter as any, monaco);

//       if (containerRef.current) {
//         monaco.editor.create(containerRef.current, {
//           value: 'const a = 1',
//           language: 'javascript',
//           theme: 'vitesse-dark',
//         });
//       }
//     };

//     initializeEditor();
//   }, []);

//   return <div ref={containerRef} id="container"></div>;
// };


export default function TestPage() {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleEditorDidMount = (editor: editor.IStandaloneCodeEditor, monaco: Monaco) => {
    // here is the editor instance
    // you can store it in \`useRef\` for further usage
    editorRef.current = editor;
  }

  // const containerRef = useRef<HTMLElement>(null);

  // useEffect(() => {
  //   const highlighter = getHighlighter({
  //     themes: ['vitesse-dark', 'vitesse-light'],
  //     langs: ['javascript', 'typescript', 'vue'],
  //   }).then((highlighter) => {
  
      
  //     monaco.languages.register({ id: 'vue' });
  //     monaco.languages.register({ id: 'typescript' });
  //     monaco.languages.register({ id: 'javascript' });
      
  //     shikiToMonaco(highlighter as any, monaco);
      
  //     if (containerRef.current) {
  //       monaco.editor.create(containerRef.current, {
  //         value: 'const a = 1',
  //         language: 'javascript',
  //         theme: 'vitesse-dark',
  //       });
  //     }
  //   });
  // }, []);

  return (
    <div className="w-full h-[calc(100vh)] flex flex-row justify-center">
      <ScrollArea className="w-full">
        <div className="flex flex-row justify-center pt-10">
          <div className="max-w-[85vw] md:max-w-[70vw] lg:max-w-[45vw]">
            {/* <div ref={containerRef} id="container"></div> */}
            {/* <Editor/> */}
            {/* <CodeEditor/> */}
            <Editor 
              height="90vh" 
              width="45vw"
              // className="border-2 border-red-500"
              theme="vs-dark"
              defaultLanguage="javascript" 
              defaultValue="// some comment"
              onMount={handleEditorDidMount as OnMount} 
            />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

// export CodeEditor;
`;

export default function TestPage() {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleEditorDidMount = (editor: editor.IStandaloneCodeEditor, monaco: Monaco) => {
    // here is the editor instance
    // you can store it in `useRef` for further usage
    editorRef.current = editor;
  }

  return (
    <div className="w-full h-[calc(100vh)] flex flex-row justify-center">
      <ScrollArea className="w-full">
        <div className="flex flex-row justify-center pt-10">
          <div className="max-w-[85vw] md:max-w-[70vw] lg:max-w-[45vw]">
            <Editor
              height="90vh" 
              width="45vw"
              theme="vs-dark"
              defaultLanguage="typescript" 
              defaultValue={TEST_CODE}
              onMount={handleEditorDidMount as OnMount} 
            />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

// export CodeEditor;