import { useEffect } from "react";

// Updates the height of a <textarea> when the value changes.
const useAutosizeTextArea = (
  textAreaRef: HTMLTextAreaElement | null,
  value: string,
  onFinish?: (height : number) => void,
) => {
  useEffect(() => {
    if (textAreaRef) {
      // We need to reset the height momentarily to get the correct scrollHeight for the textarea
      textAreaRef.style.height = "0px";
      // const scrollHeight = Math.min(textAreaRef.scrollHeight, maxHeight);
			
      // We then set the height directly, outside of the render loop
      // Trying to set this with state or a ref will product an incorrect value.
      textAreaRef.style.height = textAreaRef.scrollHeight + "px";
      if (onFinish) onFinish(textAreaRef.scrollHeight);
    }
  }, [textAreaRef, value, onFinish]);
};

export default useAutosizeTextArea;
