"use server";
import { getHighlighter } from 'shiki';
import { shikiToMonaco } from '@shikijs/monaco';
import * as monaco from 'monaco-editor-core';


export default async function initializeEditor(domElement : HTMLElement) {
  const highlighter = getHighlighter({
    themes: ['vitesse-dark', 'vitesse-light'],
    langs: ['javascript', 'typescript', 'vue'],
  }).then((highlighter) => {
    
    monaco.languages.register({ id: 'vue' });
    monaco.languages.register({ id: 'typescript' });
    monaco.languages.register({ id: 'javascript' });
    
    shikiToMonaco(highlighter as any, monaco);
    
    monaco.editor.create(domElement, {
      value: 'const a = 1',
      language: 'javascript',
      theme: 'vitesse-dark',
    });
  });
};