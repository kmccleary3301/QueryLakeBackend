export default function craftUrl(host : string, parameters : object) {
  // Use a placeholder and replacement because otherwise it complains about the URL not being valid.
  const url = new URL("http://t.c/");
  const stringed_json = JSON.stringify(parameters);
  url.searchParams.append("parameters", stringed_json);
  const return_url = url.toString().replace("http://t.c/", host);
  // console.log("Returning URL:", return_url);
  return return_url;
}