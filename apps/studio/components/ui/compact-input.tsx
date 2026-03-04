import { DetailedHTMLProps, HTMLInputTypeAttribute, InputHTMLAttributes, forwardRef, useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import * as Icon from 'react-feather';
import { AspectRatio } from './aspect-ratio';

const CompactInput = forwardRef(({
  className,
  type = "text",
  placeholder,
  backgroundColor = undefined,
  hideText = false,
  ...props
}: {
  className?: string;
  type?: HTMLInputTypeAttribute;
  placeholder?: string;
  backgroundColor?: string;
  hideText?: boolean;
} & DetailedHTMLProps<InputHTMLAttributes<HTMLInputElement>, HTMLInputElement>, ref) => {
  const [isFocused, setIsFocused] = useState(false);
	const [placeHolderMoved, setPlaceHolderMoved] = useState(
		(props.value && props.value !== "") ? true : 
		(props.defaultValue && props.defaultValue !== "") ? true :
		false
	);
	const [hidden, setHidden] = useState((type === "password") ? true : hideText);
	const buttonAvailable = (type === "password") ? true : false;


	// const height = className ? className.match(/h-(^[\s]+)/)?.[1] : "h-10";

	const animatePlaceholderStyle = (condition : boolean) => {
		return (condition) ? 
			{ y: 'calc(0% - 0.75em - 0px)', fontSize: '0.75em', paddingLeft: '0.2rem' } : 
			{ y: 'calc(50% - 1em + 2px)', fontSize: '1em', paddingLeft: '0.5rem'}
	}

  return (
    <div 
			className={cn(
				`bg-background flex h-10 rounded-md border border-input ring-offset-background file:border-0`,
        "file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:ring-2",
        "focus-visible:ring-ring focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50",
				className, 
				"flex flex-col-reverse mx-0 px-0",
				isFocused ? "border-primary" : "",
			)}
      style={
        (backgroundColor) ? {backgroundColor: backgroundColor} : {}
      }>
			<div className='flex flex-row py-0 px-0 h-full rounded-md w-auto max-w-full'>
				<div className='flex-grow overflow-hidden px-0'>
					<input
						className={
							`${buttonAvailable ? "w-full" : "w-full"} p-0 px-3 bg-transparent rounded-md ring-transparent focus-visible:ring-transparent focus-visible:border-none focus:outline-none h-full overflow-hidden`
						}
						// type={type}
						type={!hidden ? "text" : type}
						{...props}
						onFocus={(e) => {
							setIsFocused(true);
							setPlaceHolderMoved(true);
							// const currentText = e.target.value;
							// console.log("FOCUS:", e, e.target.value);
							if (props && props.onFocus) props.onFocus(e);
						}}
						onBlur={(e) => {
							setIsFocused(false);
							const currentText = e.target.value;
							if (currentText === "") setPlaceHolderMoved(false);
							// console.log("BLUR:", e);
							if (props && props.onBlur) props.onBlur(e);
						}}
						spellCheck={false}
						// hidden={true}
					/>
				</div>
				{buttonAvailable && (
					<div className='p-0 m-0 h-full rounded-l-none rounded-md'>
					<Button variant={"ghost"} type="button" className={cn(
						"border-[3.5px] border-transparent h-full w-full py-0 m-0 px-2",
						(isFocused ? "rounded-md" : "rounded-sm"),
						"rounded-l-none"
					)} onClick={() => {
						setHidden(!hidden);
					}}>
						{/* <AspectRatio ratio={1} className="flex items-center justify-center p-0 m-0 h-full"> */}
						{(hidden) ? <Icon.EyeOff className='w-4 h-4 text-muted-foreground' /> : <Icon.Eye className='w-4 h-4 text-muted-foreground' />}
						{/* </AspectRatio> */}
					</Button>
					</div>
				)}
			</div>
			<div className="h-0 pointer-events-none placeholder-inherit">
				<div className={cn(
					"h-10 px-3 py-0",
					className,
					"pl-0 text-inherit placeholde-inherit w-auto"
				)} onFocus={()=>{console.log("Focus Called!")}}>
					<motion.div
						className='h-full'
						initial={animatePlaceholderStyle(placeHolderMoved)}
						animate={animatePlaceholderStyle(placeHolderMoved)}
						transition={{ duration: 0.2 }}
					>
						<span className={cn(`placeholder bg-background px-1 rounded-sm`, isFocused?"text-primary":"text-muted-foreground")} style={
              (backgroundColor) ? {backgroundColor: backgroundColor} : {}
            }>{placeholder}</span>
					</motion.div>
				</div>
			</div>
		</div>
  );
})

CompactInput.displayName = "CompactInput";

export default CompactInput;
