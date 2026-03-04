export const toVariableName = (str : string) => {
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

export function getValueFromPath(obj: Record<string, any>, path: string[]): any {
	let current = obj;
	for (let i = 0; i < path.length; i++) {
		if (current[toVariableName(path[i])] === undefined) {
			return undefined;
		}
		current = current[toVariableName(path[i])];
	}
	return current;
}