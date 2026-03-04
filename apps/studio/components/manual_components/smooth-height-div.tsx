import { DetailedHTMLProps, HTMLAttributes, useCallback, useEffect, useRef } from "react";
import { motion, useAnimation } from "framer-motion";
import useResizeObserver from '@react-hook/resize-observer';
import { cn } from "@/lib/utils";

export default function SmoothHeightDiv({
  children,
  style,
  className="",
  ...props
}: DetailedHTMLProps<HTMLAttributes<HTMLDivElement>, HTMLDivElement>) {
  const contentRef = useRef<HTMLDivElement>(null);
  const motionDivAnim = useAnimation();
  const motionDiv = useRef<HTMLDivElement>(null);
  const motionDivHeight = useRef<number>(0);

  useEffect(() => {
    motionDivAnim.stop()
    motionDivAnim.start({
      height: "auto"
    })
  }, [motionDivAnim])

  // const onResize = useCallback((entry : ResizeObserverEntry) => {
  //   const height = entry.contentRect.height;
  //   // console.log("height: ", height);

  //   // if (motionDiv.current) {
  //   //   motionDiv.current.style.height = `${height}px`;
  //   // }
  //   // console.log("motionDiv Height at time of call: ", motionDivHeight.current);


  //   // motionDivAnim.stop();
  //   // motionDivAnim.start({
  //   //   height: height,
  //   //   transition: { duration: 0.3, ease: [0.23, 1, 0.32, 1] }
  //   // });
  // }, [contentRef, motionDivAnim, motionDivHeight])

  // useResizeObserver(contentRef, onResize);
  // useResizeObserver(motionDiv, (entry) => {
  //   // console.log("motionDiv Height: ", entry.contentRect.height);
  //   motionDivHeight.current = entry.contentRect.height;
  // });


  return (
    <motion.div
      ref={motionDiv}
      style={{ 
      // overflow: "hidden",
        maxWidth: "100%",
        overflowY: "hidden",
        // ...style
      }}
      initial={{ height: "auto" }}
      animate={motionDivAnim}
      // className="overflow-x-hiddem"
      // className={cn(className, "max-w-full")}
      // transition={{ 
      //   duration: 0.3,
      //   ease: [0.04, 0.62, 0.23, 0.98]
      // }}
    >
      <div ref={contentRef} className="" style={{maxWidth: "100%",}}>
        <div {...props} style={style} className={className}>
          {children}
        </div>
      </div>
    </motion.div>
  );
}

