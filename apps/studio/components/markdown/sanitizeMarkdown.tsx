// function stripNewlinesFromDollarSegments(inputString : string) {
//   return inputString.replace(/(?:[^\\]|^)\$\$([\s\S]*?)\$\$(?:[^\\]|$)/g, (match, p1) => `$$${p1.replace(/\n/g, '')}$$`);
// }

function stripNewlinesFromDollarSegments(inputString : string) {
  return inputString.replace(/\$\$([\s\S]*?)\$\$/g, (match, p1) => ` $$ ${p1.replace(/\n/g, '')} $$ `);
}

function stripNewlinesFromSlashSegments(inputString : string) {
  return inputString.replace(/(?:[^\\]|^)\\\[([\s\S]*?)\\\](?:[^\\]|$)/g, (match, p1) => `\\[ ${p1.replace(/\n/g, '')} \\]`);
}

export default function sanitizeMarkdown(input_string : string) {
  
  let ret = input_string;
  ret = ret.replace(/^[\s]*/, "");
  ret = stripNewlinesFromDollarSegments(ret);
  ret = stripNewlinesFromSlashSegments(ret);

  let input_split = ret.split("\n");
  const line_is_table_row = Array(input_split.length).fill(false);
  const line_is_whitespace = Array(input_split.length).fill(false);

  const line_action = Array(input_split.length).fill(0); //0 for nothing, 1 for newline, 2 for deletion, 3 for header line insertion, 4 for both 3 and 2.

  const table_column_counts = [];

  // console.log("Matching test divider");
  // console.log("| --- | --- |".match(/^([\s]*(\|[\s]*[-]+[\s]*)+\|[\s]*)$/));

  for (let i = 0; i < input_split.length; i++) {
    const table_match = input_split[i].match(/^([\s]*\|([^\n])+\|[\s]*)$/);
    const whitespace_match = input_split[i].match(/^([\s]*)$/);
    if (whitespace_match !== null) { line_is_whitespace[i] = true; }
    if (table_match !== null) {
      if (i > 1 && line_is_table_row[i-1] && !line_is_table_row[i-2]) {
        if ( input_split[i].match(/^([\s]*(\|[\s]*[-]+[\s]*)+\|[\s]*)$/) === null) { //Check if second row is already a divider.
          line_action[i-1] = 3;
          table_column_counts.push(input_split[i].split("|").length-1);
        }
      }
      if (i > 0 && !line_is_table_row[i-1]) {
        if (!line_is_whitespace[i-1]) {
          line_action[i] = 1; //Insert empty line before start of table.
        }
        
      } else { 
        if (i > 1 && line_is_whitespace[i-1] && line_is_table_row[i-2]) {
          line_action[i-1] = (line_action[i-1] === 3)?4:2; //Delete empty line between table rows.
        }
      }

      line_is_table_row[i] = true;
    } else {
      if (i > 0 && line_is_table_row[i-1] && !line_is_whitespace[i]) {
        line_action[i] = 1; //Insert empty line after end of table.
      }
    }
  }

  let table_divider_push_index = 0;

  for (let i = line_action.length-1; i >= 0; i--) {
    if (line_action[i] === 2 || line_action[i] === 4) {
      input_split.splice(i, 1);
    } else if (line_action[i] === 1) {
      input_split = [...input_split.slice(0, i), "", ...input_split.slice(i)]; 
    }
    if (line_action[i] === 3 || line_action[i] === 4) {
      let fill_sequence = Array(table_column_counts[table_divider_push_index]).fill("| --- ").join("");
      fill_sequence = fill_sequence.slice(0, fill_sequence.length - 5);
      input_split = [...input_split.slice(0, i+1), fill_sequence, ...input_split.slice(i+1)]; 
      table_divider_push_index += 1;
    }
  }
  ret = input_split.join("\n");

  // console.log(ret);
  // console.log("Input", [input_string], "Output", [ret]);
  // return stripNewlinesFromDollarSegments(ret);
  return ret;
}