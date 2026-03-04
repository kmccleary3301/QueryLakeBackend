const fs = require("fs");
const path = require("path");

const toVariableName = (str) => {
  // Remove non-alphanumeric characters and convert to lower case
  str = str.replace(/\.md$/, "")
  str = str.replaceAll(" ", "_")
  str = str.replaceAll(/[^a-zA-Z0-9_]/g, '').toLowerCase();

  // Split by spaces
  let words = str.split(' ');

  // // Remove any word that starts with a number
  // words = words.filter(word => isNaN(word[0]));

  // Convert to camelCase
  for (let i = 1; i < words.length; i++) {
    words[i] = words[i].charAt(0).toUpperCase() + words[i].slice(1);
  }

  return words.join('').replace(/[\_]+$/, '').replace(/^[\_]+/, '').replaceAll(/[\_]+/g, '_');
}

const create_export_content = (full_path, export_name) => {

  // const subContent = iterateDirectory(full_path);

  let folder_structure = {};

  const file_names = fs.readdirSync(full_path);

  let imports = "";
  let json_created = "{\n";
  for (let i = 0; i < file_names.length; i++) {
    const file_name = file_names[i];

    if (file_name === ".obsidian") {
      continue;
    }
    const is_md = path.extname(file_name) === ".md";
    const statsPointer = fs.statSync(path.join(full_path, file_name));
    const variableName = toVariableName(file_name);
    if (is_md && statsPointer.isFile()) {
      imports += `import ${variableName} from "./${variableName}"; //MD IDENTIFIED\n`;
    } else if (statsPointer.isDirectory() && (file_name !== ".obsidian")) {
      imports += `import ${variableName} from "./${file_name}/__export__";\n`;
    }
    json_created += `\t\"${variableName}\": ${variableName}${(i === (file_names.length-1))?'':','}\n`;
  }

  return `
${imports}

const ${export_name} = ${json_created}};

export default ${export_name};
`;
}

const createFolderStructure = (directory_in) => {
  const directory = (Array.isArray(directory_in))?directory_in:[directory_in];
    

  var folder_structure = {}, aliases = {}, reverse_lookup = {};
  const fileNames = fs.readdirSync(path.join(...directory));
  fileNames.forEach((fileName) => {
    const fullPath = path.join(...directory, fileName);
    // Ignore .obsidian directory
    if (fileName === ".obsidian") return;
    const stats = fs.statSync(fullPath);
    if (stats.isDirectory()) {
      const [sub_folder_structure, sub_aliases, sub_reverse_lookup] = createFolderStructure([...directory, fileName]);
      const sub_aliases_copy = JSON.parse(JSON.stringify(sub_aliases));
      aliases = {...sub_aliases_copy, ...aliases};

      reverse_lookup = {...sub_reverse_lookup, ...reverse_lookup};
      folder_structure[fileName] = JSON.parse(JSON.stringify(sub_folder_structure));
    } else if (stats.isFile() && path.extname(fileName) === ".md") {
      

      const filename_no_ext = fileName.replace(/\.md$/, "");
      folder_structure[`${filename_no_ext}`] = null;

      const full_path_split = directory.slice();
      full_path_split.push(filename_no_ext)
      const full_path_variable_names = full_path_split.map((path_part) => toVariableName(path_part));
      
      reverse_lookup[
        `${full_path_variable_names.slice(1, full_path_variable_names.length).join("/")}`
      ] = full_path_split.slice(1);

      // console.log(full_path_split);
      // console.log(full_path_variable_names);

      for (let i = 0; i < full_path_split.length; i++) {
        const path_alias = full_path_split.slice(i, full_path_split.length).join("/");
        aliases[`${path_alias}`] = `${full_path_variable_names.slice(1).join("/")}`;
      }

    }
  });
  // console.log(aliases);
  return [folder_structure, aliases, reverse_lookup];
};

const escape_replace = [
	[/\%/g, "%"],
	[/\\\*/g, "*"],
	[/\\\`/g, "`"],
	[/\\\$/g, "$"],
  [/\\\[/g, "["],
  [/\\\]/g, "]"]
]

function escape_text(text) {
	for (let i = 0; i < escape_replace.length; i++) {
		text = text.replaceAll(escape_replace[i][0], `-%${i}%-`);
	}
	return text;
}

function unescape_text(text) {
	for (let i = escape_replace.length - 1; i >= 0; i--) {
		text = text.replaceAll(`-%${i}%-`, escape_replace[i][1]);
	}
	return text;
}

const aliasLookupAndReplace = (text, alias_lookup) => {
  text = escape_text(text);

  const all_md_patterns = /(\[\[.*?\]\])/;
  let match = text.match(all_md_patterns);
  let index = 0;
  const string_segments = [];
  if (match === null) {
      string_segments.push(unescape_text(text));
  }
  while (match !== null && match !== undefined && match.index !== undefined) {
    // console.log("match:", match);
    if (match.index > 0) {
      // console.log()
      string_segments.push(unescape_text(text.slice(index, index+match.index)))
    }
    if (match[0].length > 4 && match[0].slice(0, 2) === "[[") {
      const inner_text = match[0].slice(2, match[0].length-2);
      if (alias_lookup[inner_text] !== undefined) {
        const lookup_path = alias_lookup[inner_text].split("/");
        string_segments.push(`[${inner_text}](/docs/${alias_lookup[inner_text]})`);
      } else {
        string_segments.push(unescape_text(match[0]));
      }
    } else {
      string_segments.push(unescape_text(match[0]));
    }
    // if (match === undefined) {}
    const new_index = index+match[0].length+match.index;
    const new_match = text.slice(new_index).match(all_md_patterns);
    if (new_match === null && new_index < text.length) {
      string_segments.push(unescape_text(text.slice(new_index)));
    }
    
    match = new_match;
    index = new_index;
  }
  return string_segments.join("");

  
}


const iterateDirectory = (directory, alias_lookup) => {

  const fileNames = fs.readdirSync(directory);
  let content = [];

  fileNames.forEach((fileName) => {
    const fullPath = path.join(directory, fileName);

    // Ignore .obsidian directory
    if (fileName === ".obsidian") {
      return;
    }

    const stats = fs.statSync(fullPath);

    

    if (stats.isDirectory()) {
      const subContent = iterateDirectory(fullPath, alias_lookup);
      // folder_structure[`${fileName}`] = `${fileName}`;

      const exportContent = create_export_content(fullPath, toVariableName(fileName));
      
      fs.writeFileSync(path.join("public/cache", fullPath, "__export__.tsx"), exportContent);
      content = content.concat(subContent);
    } else if (stats.isFile() && path.extname(fileName) === ".md") {

      const fileContents = fs.readFileSync(fullPath, "utf8");
      
      const matterResult = {
        content: aliasLookupAndReplace(fileContents, alias_lookup)
      };
      const post = {
        slug: fileName.replace(/\.md$/, ""),
        ...matterResult,
      };
      content.push(post);

      const postFileContents = `
const ${toVariableName(fileName)} = ${JSON.stringify(post, null, '\t')};

export default ${toVariableName(fileName)};
`;
      // const new_path = path.join("public/cache", directory, toVariableName(fileName) + ".tsx");
      // const tsxFilePath = path.join("public/cache", fullPath.replace(/\.md$/, ".tsx"));
      
      const tsxFilePath = path.join("public/cache", directory, toVariableName(fileName) + ".tsx");
      const tsxDirPath = path.dirname(tsxFilePath);

      // Create the directory path if it doesn't exist
      if (!fs.existsSync(tsxDirPath)) {
        fs.mkdirSync(tsxDirPath, { recursive: true });
      }

      fs.writeFileSync(tsxFilePath, postFileContents);
    }
  });

  return content;
};

// Clear the cache folder if it exists
try {
  fs.rmSync("public/cache", { recursive: true });
} catch (e) {
  console.error(`Error while deleting cache directory. ${e}`);
}

// Create the cache folder
try {
  fs.mkdirSync("public/cache", { recursive: true });
} catch (e) {
  console.error(`Error while creating cache directory. ${e}`);
}

const READ_DIRECTORY = "documentation";

let MainExportContent = create_export_content(READ_DIRECTORY, "allDocs");

const [ MainFolderStructure, MainFolderAliases, MainReverseLookup ] = createFolderStructure(READ_DIRECTORY);


iterateDirectory(READ_DIRECTORY, MainFolderAliases);
// console.log(MainFolderStructure);


MainExportContent += `

export const folder_structure = ${JSON.stringify(MainFolderStructure, null, '\t')};

export const reverse_lookup = ${JSON.stringify(MainReverseLookup, null, '\t')};

export const folder_structure_aliases = ${JSON.stringify(MainFolderAliases, null, '\t')};
`;

fs.writeFileSync(path.join("public/cache", READ_DIRECTORY, "__all-documents__.tsx"), MainExportContent);

console.log("Posts cached.");