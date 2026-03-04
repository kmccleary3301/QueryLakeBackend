const fs = require("fs");
const path = require("path");



const read_themes = () => {
  
}


const READ_DIRECTORY = "themes";

let MainExportContent = create_export_content(READ_DIRECTORY, "allDocs");

const [ MainFolderStructure, MainFolderAliases, MainReverseLookup ] = createFolderStructure(READ_DIRECTORY);


iterateDirectory(READ_DIRECTORY, MainFolderAliases);
// console.log(MainFolderStructure);


MainExportContent += `
export const SYSTEM_THEMES = ${JSON.stringify(parsed_themes, null, '\t')};
`;

fs.writeFileSync(path.join("public/cache", READ_DIRECTORY, "all_themes.tsx"), MainExportContent);

console.log("Posts cached.");