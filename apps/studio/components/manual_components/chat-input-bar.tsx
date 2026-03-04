import { useState, useRef, useImperativeHandle} from 'react'
// import '@/App.css'
import { Button } from '@/components/ui/button'
import { PaperPlaneIcon } from '@radix-ui/react-icons'
import AdaptiveTextArea from '@/components/ui/adaptive-text-area'
import ChatInput from '@/components/ui/chat-input'

type ChatBarInputProps = {
	onMessageSend?: (message: string) => void,
  handleDrop?: (files: File[]) => void,
  fileDropEnabled?: boolean,
  chatEnabled?: boolean,
  onHeightChange?: (height : number) => void
}

export default function ChatBarInput(props : ChatBarInputProps) {
  const [userInput, setUserInput] = useState("");
  const [filesQueued, setFilesQueued] = useState(false);

  
  const texteAreaRef = useRef<HTMLTextAreaElement>(null)
  useImperativeHandle(texteAreaRef, () => texteAreaRef.current!);

  const passProps = (props.handleDrop !== undefined)?{onDrop: (files : File[]) => {
    if (props.handleDrop) {
      setFilesQueued(true);
      props.handleDrop(files);
    }
  }}:{};

  return (
    <div className='flex flex-row justify-center pt-0'>
      <div className='w-[60vw] flex flex-row justify-center'>
        <ChatInput
          {...passProps}
          // value={userInput}
          // setValue={setUserInput}
          onSubmission={(value : string) => {
            if (props.onMessageSend) props.onMessageSend(value);
          }}
        />
        
        <div className='flex h-auto pl-[10px] flex-col justify-center'>
          <Button variant="secondary" type="submit" size="icon" disabled={(userInput.length < 1 && !filesQueued)} onClick={()=>{
            if (props.onMessageSend) props.onMessageSend(userInput);
            setUserInput("");
            setFilesQueued(false);
          }}>
            <PaperPlaneIcon className="h-4 w-4 text-primary" />
          </Button>
        </div>
      </div>
    </div>
  )
}
