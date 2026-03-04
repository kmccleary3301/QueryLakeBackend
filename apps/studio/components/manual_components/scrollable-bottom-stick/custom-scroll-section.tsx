"use client";
import { useState, useRef, useEffect, useLayoutEffect, useCallback } from "react";
import * as Icon from 'react-feather';
import { Button } from "@/components/ui/button";
import useResizeObserver from '@react-hook/resize-observer';
import { ScrollBar } from "@/components/ui/scroll-area";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"
import { cn } from "@/lib/utils";


type TestScrollSectionProps = {
	children: React.ReactNode,
}


const useSize = (target : React.RefObject<HTMLDivElement>) => {
  const [size, setSize] = useState<DOMRect>()

  useLayoutEffect(() => {
		if (target.current !== null) {
			setSize(target.current.getBoundingClientRect())
		}
  }, [target])

  // Where the magic happens
  useResizeObserver(target, (entry) => setSize(entry.contentRect))
  return size
}

export default function ScrollSection({
  horizontal = false,
  scrollToBottomButton = false,
	scrollBar = true,
	className = "",
	innerClassName = "",
	children,
} : { 
  horizontal?: boolean,
  scrollToBottomButton?: boolean,
	scrollBar?: boolean,
	className?: string,
	innerClassName?: string,
	children: React.ReactNode,
}) {

	// Add a new state to keep track of whether there's overflow
	const [isOverflowing, setIsOverflowing] = useState(false);
	const [animateScroll, setAnimateScroll] = useState(true);
	const scrollDiv = useRef<HTMLDivElement>(null);
	const oldScrollValue = useRef(0);
  const oldScrollHeight = useRef(0);
	// const [chatBarHeightString, setChatBarHeightString] = useState(40);
	const observer = useRef<ResizeObserver>();	
	

	const interiorDiv = useRef<HTMLDivElement>(null);
	const interiorDivSize = useSize(interiorDiv);
	const [smoothScroll, setSmoothScroll]	= useState(true);
	

	const scrollToBottomHook = useCallback(({
    smooth = undefined,
  }:{
    smooth?: boolean
  }) => {
		// console.log("Scrolling to bottom");
		if (scrollDiv.current !== null) {
			scrollDiv.current.scrollTo({
				top: scrollDiv.current.scrollHeight,
        behavior: ((smooth !== undefined)?smooth:smoothScroll)?"smooth":"instant" as ScrollBehavior
      });
    }
  }, [smoothScroll, scrollDiv]);


	// useEffect(() => {
	// 	setTimeout(() => { setSmoothScroll(true); }, 1000)
	// }, []);

	// useEffect(() => {
	// 	// console.log("Resize monitor triggered");
	// 	if (animateScroll) {
	// 		scrollToBottomHook({});
	// 	}
	// }, [interiorDivSize, scrollToBottomHook]);

	useEffect(() => {
		// console.log("interiorDivSize Height changed to:", interiorDivSize?.height);
		if (animateScroll && interiorDivSize?.height !== undefined) {
			scrollToBottomHook({smooth: false});
		}

    return () => {
      if (observer.current) {
        observer.current.disconnect();
      }
    }
	}, [interiorDivSize?.height, animateScroll, scrollToBottomHook]);


	useEffect(() => {
		const div = interiorDiv.current;
		if (!div) return;
		
		if (animateScroll) {
			observer.current = new ResizeObserver(() => {
				if (animateScroll) {
					scrollToBottomHook({});
				}
			});
		} else if (observer.current) {
			observer.current.unobserve(div)
		}
	}, [animateScroll, scrollToBottomHook]);

	return (
		<>
			<ScrollAreaPrimitive.Root
				className={cn(`h-full w-full overflow-${horizontal?"x":"y"}-hidden`, className)}
			>
				<ScrollAreaPrimitive.Viewport
					ref={scrollDiv}
					id="scrollAreaPrimitive1" 
					className="h-full w-full rounded-[inherit]"
					// onChange={() => {
					// 	// console.log("Change called");
					// 	if (animateScroll && scrollToBottomButton) {
					// 		scrollToBottomHook({});
					// 	}
					// }}
					onScroll={(e) => {
						const isNowOverflowing = e.currentTarget.scrollHeight > e.currentTarget.clientHeight;
						if (isOverflowing !== isNowOverflowing) {
							setIsOverflowing(isNowOverflowing);
						}

						if (animateScroll && e.currentTarget.scrollTop < oldScrollValue.current - 3 && 
								e.currentTarget.scrollHeight === oldScrollHeight.current &&
								isNowOverflowing) {
							// console.log("Unlocking");
							setAnimateScroll(false);
						} else if (!animateScroll && Math.abs( e.currentTarget.scrollHeight - (e.currentTarget.scrollTop + e.currentTarget.clientHeight)) < 5) {
							// console.log("Locking Scroll");
							scrollToBottomHook({});
							setAnimateScroll(true);
						}
						oldScrollValue.current = e.currentTarget.scrollTop;
						oldScrollHeight.current = e.currentTarget.scrollHeight;
					}}
				>
					<ScrollAreaPrimitive.Viewport id="scrollAreaPrimitive2" className="flex flex-col" ref={interiorDiv}>
						<div className="flex flex-row w-full">
							<div className={cn(`flex flex-col`, innerClassName)}>
								{children}
							</div>
						</div>
					</ScrollAreaPrimitive.Viewport>
				</ScrollAreaPrimitive.Viewport>
				{/* <div className="w-full h-[20px] bg-purple-500"/> */}
				{scrollToBottomButton && (
					<div id="InputBox" className="flex flex-row justify-center h-0 pb-0">
						<div className="absolute h-0 flex flex-grow flex-col justify-end">
							<div className="bg-none flex flex-col justify-around pt-[10px] pb-0">
								<div id="scrollButton" className="bg-transparent flex flex-row justify-end pb-[10px]">
									{(animateScroll === false && isOverflowing) && (
										<Button onClick={() => {
											// setAnimateScroll(true);
											if (scrollDiv.current !== null) {
												setAnimateScroll(true);
												scrollToBottomHook({});
											}
										}} className="rounded-full shadow-base shadow-secondary z-5 p-0 w-10 h-10 items-center bg-input hover:bg-input/60 active:bg-input/40" variant={"secondary"}>
											<Icon.ChevronDown className="text-primary w-[50%] h-[50%]" />
										</Button>
									)}
								</div>
							</div>
						</div> 
					</div>
				)}
				<ScrollBar className={`opacity-${scrollBar?"100":"0"}`} orientation={horizontal?"horizontal":"vertical"}/>
    		{/* <ScrollAreaPrimitive.Corner /> */}
			</ScrollAreaPrimitive.Root>
		</>
  );
}
