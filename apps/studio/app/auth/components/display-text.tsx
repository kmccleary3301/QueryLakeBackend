"use client";
// import { useEffect } from "react";
// import { stagger} from "framer-motion";
import { TextGenerateEffect } from "./text-generate-effect";
import { motion, stagger, useAnimate } from "framer-motion";
import LoginBox from "./login-box";
import React, { useState, useEffect } from 'react';
import { cn } from "@/lib/utils";
import { useThemeContextAction } from "@/app/theme-provider";

const LSULogoSVG = [
  "m340.55 617.76v-0.64375-61.648-0.64375h0.64375 55.584 0.64375v0.64375 12.483 0.64375h-0.64375-33.633v48.522 0.64375h-0.64375-21.308-0.64375z",
  "m78.627-0.002c-10.501 0-19.044-6.4421-19.044-14.361v-23.348-0.64375h0.64375 43.978v-6.0609c0-2.5761-2.9439-4.6719-6.5641-4.6719h-37.414-0.64375v-0.64376-12.562-0.64374h0.64375 47.459c10.544 0 19.12 6.4421 19.12 14.361v23.428 0.64375h-0.64375-43.978v5.9812c0 2.5336 2.9688 4.6734 6.4859 4.6734h37.492 0.64375v0.64374 12.561 0.64375h-0.64375-47.536z",
  "m95.441 0v-0.64375-43.77c0-2.6199-2.9452-4.75-6.5641-4.75h-15.612v48.367 0.64375h-0.64376-21.086-0.64375v-0.64375-61.495-0.64374h0.64375 47.387c10.503 0 19.047 6.4421 19.047 14.361v47.931 0.64375h-0.64375-21.241-0.64375z"
];

const MoveLeftWrapper = ({ children, className = "" } : { children: React.ReactNode, className? : string }) => {
	return (
		<div className={cn("w-full items-center justify-center flex flex-row", className)}>
			{/* <motion.div
				id="text segment 1"
				className="flex-row"
				initial={{ marginLeft: "50%" }} // start from center
				animate={{ marginLeft: "50%" }} // move to start
				transition={moveLeftAnimation}
			>
				<div className="inline-block">
					<motion.div
						initial={{ marginLeft: "-50%" }} // start from center
						animate={{ marginLeft: "-50%" }} // move to start
						transition={moveLeftAnimation}
					> */}
						{children}
					{/* </motion.div>
				</div>
			</motion.div> */}
		</div>
	)
}

export const QueryLakeDisplay = ({ children }:{ children: React.ReactNode }) => {
	const moveLeftAnimation = { type: "spring", duration: 1, bounce: 0, delay: 4.5 };
	const padLeft = "2.5%";
	const LSULogoSize = 0.5;
  const { theme, generateStylesheet } = useThemeContextAction();


	return (
		<>
			<div 
				id="spacing div" 
				className="h-full w-full flex flex-col items-center justify-around overflow-x-scroll scrollbar-hide"
        style={generateStylesheet(theme.dark)}
			>
				{/* <div /> */}
				<div className="w-full pt-[20px] pb-[20px] text-white">
					<MoveLeftWrapper className="pb-[0px]">
						<TextGenerateEffect
							initialDelay={0.5}
							staggerDelay={0.05}
							duration={2}
							spring={true}
							words="QueryLake"
							// words="Test"
							className="inline-block text-primary text-2xl md:text-4xl lg:text-6xl font-bold"
						/>
					</MoveLeftWrapper>
					<MoveLeftWrapper className="pt-[2px] pb-[8px]">
						<TextGenerateEffect
							initialDelay={2.3}
							staggerDelay={0.05}
							words="An AI platform for everyone"
							// words="Test"
							className="inline-block text-base md:text-base mt-4 font-normal inter-var"
						/>
					</MoveLeftWrapper>
					<LoginBox delay={5}>{children}</LoginBox>
					<MoveLeftWrapper className="pt-[14px]">
						<motion.div
							initial={{ opacity: 0 }} 
							animate={{ opacity: 1 }} 
							transition={{ type: "spring", duration: 2.5, delay: 3.5 }}
						>
							{/* <div style={{ transform: "scale(0.3)" }}> */}
								<svg
									xmlns="http://www.w3.org/2000/svg"
									height={(80 * 0.5 * LSULogoSize).toString()}
									viewBox="0 0 250 81"
									width={(250 * 0.5 * LSULogoSize).toString()}
									scale={1}
								>
									<g transform="matrix(1.25 0 0 -1.25 -424 774.12)">
										<path d={LSULogoSVG.join(" ")} fill="currentColor" />
									</g>
								</svg>
							{/* </div> */}
						</motion.div>
					</MoveLeftWrapper>
				</div>

				
			</div>
			{/* <TextGenerateEffect staggerDelay={0.05} words="Create, deploy, and test an AI workflow in minutes." className="text-base md:text-lg mt-4 text-white font-normal inter-var text-center"/> */}
		</>
	);
};
