import { ReactNode, StyleHTMLAttributes } from 'react';
// import { Button } from "../ui/button";
// import * as Icon from 'react-feather';
// import { useRef } from 'react';

type ScrollViewBottomStickOuterProps = {
  style?: StyleHTMLAttributes<HTMLDivElement>,
  children?: ReactNode,
	scrollDiv: React.RefObject<HTMLDivElement>,
	animateScroll: boolean,
	setAnimateScroll: React.Dispatch<React.SetStateAction<boolean>>,
	oldScrollValue: React.MutableRefObject<number>,
	oldScrollHeight: React.MutableRefObject<number>,
};

export default function ScrollViewBottomStickOuter(props: ScrollViewBottomStickOuterProps) {
  return (
    <div className="scrollbar-custom mr-1 h-full overflow-y-auto" style={{
			// margin: 1,
			width: "100%",
			// height: "full",
			overflowY: "auto",
		}} 
		ref={props.scrollDiv}
		onChange={() => {
			if (props.animateScroll) {
				if (props.scrollDiv.current !== null) {
					props.scrollDiv.current.scrollTo({
						top: props.scrollDiv.current.scrollHeight,
						behavior: 'smooth'
					});
				}
			}
		}}
		onScroll={(e) => {
			if (props.animateScroll && e.currentTarget.scrollTop < props.oldScrollValue.current - 3 && e.currentTarget.scrollHeight === props.oldScrollHeight.current) {
				props.setAnimateScroll(false);
			} else if (props.animateScroll) {
				if (props.scrollDiv.current !== null) {
					props.scrollDiv.current.scrollTo({
						top: props.scrollDiv.current.scrollHeight,
						behavior: 'smooth'
					});
				}
			}
			// scrollValue.current.scrollIntoView() = 0;
			// if (!animateScroll && Math.abs( e.currentTarget.scrollHeight - (e.currentTarget.scrollTop + e.currentTarget.clientHeight)) < 5) {
			//   setAnimateScroll(true);
			// }
			if (props.animateScroll) {
				if (props.scrollDiv.current !== null) {
					props.scrollDiv.current.scrollTo({
						top: e.currentTarget.scrollHeight,
						behavior: 'smooth'
					});
				}
			}
			props.oldScrollValue.current = e.currentTarget.scrollTop;
			props.oldScrollHeight.current = e.currentTarget.scrollHeight;
			// console.log("onScroll main window: ", e.currentTarget.scrollTop, e.currentTarget.scrollHeight, e.currentTarget.clientHeight);
		}}>
			<div style={{
				display: "flex",
				flexDirection: "column",
			}}>
				{(props.children) && (
					props.children
				)}
			</div>
    </div>
  );
}