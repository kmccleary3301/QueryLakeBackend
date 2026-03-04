import { codeToHtml, codeToTokens } from "shiki";

export default async function Code({ code, lang}: { code: string, lang: string}) {
  const html = await codeToHtml(code, {
    lang: lang,
    theme: "dark-plus",
  });

  return <code dangerouslySetInnerHTML={{ __html: html }}></code>;
}