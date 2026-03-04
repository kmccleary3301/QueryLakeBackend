"use client";
import { cn } from "@/lib/utils";
import MarkdownTextSplitter from "./markdown-text-splitter";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { markdownRenderingConfig } from "./configs";

type textSegment = {
  text: string,
  containerHeight?: number,
}

type MarkdownTableProps = {
  header: textSegment[],
  rows: textSegment[][],
  unProcessedText?: string,
  fontSize?: number,
}

function MarkdownTable({
  className = "",
  header,
  rows,
  unProcessedText,
  fontSize,
  config,
}:{
  className?: string,
  header: textSegment[],
  rows: textSegment[][],
  unProcessedText?: string,
  fontSize?: number,
  config: markdownRenderingConfig
}){

  return (
    <div className={cn("not-prose mb-[1em]", className)}>
      <Table>
        <TableHeader>
          <TableRow>
            {(header).map((header : textSegment, index : number) => (
              <TableHead key={index}>{header.text}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row : textSegment[], row_index : number) => (
            <TableRow key={row_index}>
              {row.map((entry : textSegment, col_index : number) => (
                <TableCell key={col_index} style={{
                  // display: "flex",
                  // flexDirection: "row",
                  // justifyContent: "flex-start",
                  // alignItems: "flex-start",
                }}>
                  <div style={{
                    // flexShrink: 1,
                    display: "flex",
                    // flexShrink: 1,
                    // flexGrow: 1,
                    // display: "flex",
                    // flexDirection: "row",
                    justifyContent: "flex-start",
                  }}>
                    <MarkdownTextSplitter
                      text={entry.text}
                      config={config}
                    />
                  </div>
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}


export default MarkdownTable;