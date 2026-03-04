"use client";
import { ScrollAreaHorizontal } from "@/components/ui/scroll-area";
import TeX from "@matejmazur/react-katex";
import "katex/dist/katex.min.css";

type MarkdownLatexProps = {
  textSeg : {text: string},
  type: "inline" | "newline"
}

export default function MarkdownLatex(props : MarkdownLatexProps){
  const throwOnError = false;

  try {
    if (props.type === "inline") {
      return <TeX math={props.textSeg.text} />;
    } else {
      return (
        <ScrollAreaHorizontal className="max-w-full w-full">
          <TeX as={"span"} block className="word-break whitespace-pre-wrap" math={props.textSeg.text} />
        </ScrollAreaHorizontal>
      );
    }
  } catch (error) {
    return (
      <p>{props.textSeg.text}</p>
    );
  }
}