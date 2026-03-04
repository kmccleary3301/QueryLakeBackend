import { ScrollArea } from '@radix-ui/react-scroll-area';
import { ReactNode, StyleHTMLAttributes } from 'react';
// import { Button } from "../ui/button";
// import * as Icon from 'react-feather';

type ScrollViewBottomStickInnerProps = {
  showsVerticalScrollIndicator?: boolean,
  style?: StyleHTMLAttributes<HTMLDivElement>,
  children: ReactNode,
  bottomMargin: number,
};

export default function ScrollViewBottomStickInner(props: ScrollViewBottomStickInnerProps) {
  // const scrollValue = useRef(0);

  // const [autoScroll, setAutoScroll] = useState(false);



  return (
    <div className="mx-auto flex h-full flex-col">
      <div className="group relative flex">
        <div className="flex flex-row justify-center w-full">
          <div className="flex flex-col w-full items-center">
            <div className="h-20" />
            {props.children}
            
            <div style={{ height: props.bottomMargin }} />
          </div>
        </div>
      </div>
    </div>
    // <ScrollArea>
    //   {props.children}
    // </ScrollArea>
  );
}