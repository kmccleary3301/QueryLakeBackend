"use client";
import { Token, Tokens } from 'marked';
import MarkdownTextSplitter from "./markdown-text-splitter";
import MarkdownCodeBlock from "./markdown-code-block";
import MarkdownTable from "./markdown-table";
import "./prose.css"
import { cn } from "@/lib/utils";
import { markdownRenderingConfig } from "./configs";

type MarkdownMapComponentErrorProps = {
  type: string
}

function MarkdownMapComponentError(props : MarkdownMapComponentErrorProps) {
  return (
    <div className="bg-red-600 rounded-lg items-center">
      <p className="text-lg text-red-600 p-2 text-center">
        {"Unfilled Markdown Component: "+props.type}
      </p>
    </div>
  );
}

function MarkdownTypeRouter({
  className = "",
  token,
  unProcessedText,
  finished,
  config,
}:{
  className?: string,
  token: Token,
  unProcessedText: string,
  finished: boolean,
  config: markdownRenderingConfig,
}) {
  const defaultFontSize = 'text-base';

  switch (token.type) {
    case 'space':
      return (
        // <br className={cn("not-prose", className)}/>
        <></>
      );
    case 'code':
      return (
        <MarkdownCodeBlock
          className={className}
          finished={finished} 
          text={token.text} 
          lang={token.lang} 
          unProcessedText={unProcessedText}
        />
      );
    case 'heading':
      if (token.raw[0] != "#") {
        return (
          <p className={className}>
            <MarkdownTextSplitter selectable={true} className={`text-left ${defaultFontSize}`} text={token.text + unProcessedText} config={config}/>
          </p>
        );
      } else {
        switch (token.depth) {
          case 1:
            return (
              <h2 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h2>
            );
          case 2:
            return (
              <h3 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h3>
            );
          case 3:
            return (
              <h4 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h4>
            );
          case 4:
            return (
              <h5 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h5>
            );
          case 5:
            return (
              <h6 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h6>
            );
          case 6:
            return (
              <h6 className={className}>
                <MarkdownTextSplitter selectable={true} text={token.text + unProcessedText} config={config}/>
              </h6>
            );
        }
      }

    case 'table':
      if (token.type === "table") { // Literally just here to get rid of the type error.
        return (
          <MarkdownTable
            className={className}
            header={token.header} 
            rows={token.rows}
            unProcessedText={unProcessedText}
            config={config}
          />
        );
      }
    case 'hr':
      return (null);
    case 'blockquote':
      // console.log("Blockquote text:", token);
      return (
        <blockquote className={cn(className, "pl-6 flex flex-col space-y-2")}>
          {(token.tokens)?(
            <>
              {token.tokens.map((v : Token, k : number) => (
                <MarkdownTypeRouter
                  className={v.type === "list" ? "ml-[2rem]" : ""}
                  finished={finished}
                  key={k}
                  unProcessedText={""}
                  token={v}
                  config={config}
                />
              ))}
            </>
          ):(
            <MarkdownTextSplitter 
              selectable={true} 
              className={`text-left ${defaultFontSize}`} 
              text={token.text + unProcessedText}
              config={config}
            />
          )}
        </blockquote>
      );
    case 'list':
      if (token.ordered) {
        return (
          <ol className={cn("not-prose", className)}>
            {token.items.map((v : Tokens.ListItem, k : number) => (
              <MarkdownTypeRouter
                finished={finished}
                key={k}
                unProcessedText={(k === token.items.length-1)?unProcessedText:""}
                token={{...v, type: "list_item"}}
                config={config}
              />
            ))}
          </ol>
        );
      } else {
        return (
          <ul className={cn("not-prose", className)}>
            {token.items.map((v : Tokens.ListItem, k : number) => (
              <MarkdownTypeRouter
                finished={finished}
                key={k}
                unProcessedText={(k === token.items.length-1)?unProcessedText:""}
                token={{...v, type: "list_item"}}
                config={config}
              />
            ))}
          </ul>
        );
      }
    case 'list_item':
      const counter = (token.raw.match(/^([^\s]+) /) || [""])[0].trimStart().trimEnd();

      return (
        <li className={cn(className, "relative")} counter-text={counter + " "}>
          {token.tokens?.map((v : Token, k : number) => (
            <MarkdownTypeRouter
              className={v.type === "list" ? "ml-[2rem]" : ""}
              finished={finished}
              key={k}
              unProcessedText={""}
              token={v}
              config={config}
            />
          ))}
        </li>
      );
    case 'paragraph':
      const lines = (token.text + unProcessedText).split("\n") || "";
      return (
        <span className={cn(className, "")}>
          {(lines.length > 1)?(
            <>
              {lines.map((line, i) => (
                <span className="pb-1" key={i}>
                  <br/>
                  <MarkdownTextSplitter 
                    selectable={true} 
                    className={`text-left text-base text-gray-200`} 
                    text={line}
                    config={config}
                  />
                </span>
              ))}
            </>
          ):(
            <MarkdownTextSplitter 
              selectable={true} 
              className={`text-left text-base text-gray-200`} 
              text={lines[0]}
              config={config}
            />
          )}
        </span>
      );
    case 'html':
      return (null);
    case 'text':
      return (
        <span className={className}>
          <MarkdownTextSplitter 
            selectable={true} 
            className={`text-left text-base text-gray-200`} 
            text={token.text + unProcessedText}
            config={config}
          />
        </span>
      );
    default:
      return (
        <MarkdownMapComponentError type={token.type}/>
      );
  }
}


export default MarkdownTypeRouter;