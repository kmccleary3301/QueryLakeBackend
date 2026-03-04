"use client";
import { motion } from "framer-motion";

export default function LoginBox({children, delay = 0} : {children: React.ReactNode, delay?: number}) {
	return (
		<div className="flex flex-row justify-center w-full pb-0">
			<motion.div
				id="login box"
				className="overflow-hidden overflow-x-hidden"
				initial={{ height: 0 }}
				animate={{ height: "auto" }} // move to start
				transition={{ type: "spring", duration: 1, bounce: 0, delay: delay }}
			>
				<div 
					id="Login box" 
					className="pb-[20px] pt-[20px] flex flex-col items-center justify-between"
				>
					<div className="p-4 w-[320px] bg-[#09090B]/85 rounded-md border-2 border-accent">
						{children}
					</div>
				</div>
			</motion.div>
		</div>
	);
};
