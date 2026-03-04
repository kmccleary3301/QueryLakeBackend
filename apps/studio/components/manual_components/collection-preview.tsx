// import {
//   View,
//   Text,
//   useWindowDimensions,
// 	Pressable,
// 	Animated,
// 	Easing
// } from 'react-native';
// import SwitchSelector from "react-native-switch-selector";
import { useState, useEffect } from 'react';
// import { Feather } from '@expo/vector-icons';
// import AnimatedPressable from './AnimatedPressable';
// import globalStyleSettings from '../../globalStyleSettings';
import { motion, useAnimation } from "framer-motion";
import { Button } from '@/components/ui/button';
import Link from 'next/link';

type CollectionPreviewProps = {
  selectedPrior?: boolean,
	onToggleSelected?: (selected: boolean) => void,
	title: string,
	documentCount: number;
	collectionId: string,
  parentSelected: boolean,
  parentMixedSelection: boolean,
}

export default function CollectionPreview(props: CollectionPreviewProps) {
  const [selected, setSelected] = useState((props.selectedPrior)?props.selectedPrior:false);

  useEffect(()=>{
    if (props.selectedPrior)
      setSelected(props.selectedPrior);
  }, [props.selectedPrior])
	const {title, documentCount} = props;

	const selectionCircleSize = useAnimation();

	useEffect(() => {
		selectionCircleSize.start({
			width: selected?11:0,
			height: selected?11:0,

			transition: { duration: 0.1 }
		});
  }, [selected, selectionCircleSize]);

  useEffect(() => {
    if (props.parentSelected) {
      setSelected(true);
    } else if (!props.parentMixedSelection && !props.parentMixedSelection) {
      setSelected(false);
    }
  }, [props.parentSelected, props.parentMixedSelection]);

	return (
		<div className={`w-full flex flex-col`}>
			<div className="h-7 flex flex-row w-full space-x-2">
        <div className='w-5 h-5'> 
        <Button className="w-5 h-5 rounded-full bg-theme-one items-center justify-center flex p-0"
          onClick={() => {
            if (props.onToggleSelected) { props.onToggleSelected(!selected); }
            setSelected(selected => !selected);
        }}>
          <motion.div animate={selectionCircleSize} className="rounded-full bg-background"/>
        </Button>
        </div>
        <Link className='flex-grow whitespace-nowrap overflow-hidden text-ellipsis' href={`/collection/edit/${props.collectionId}`}>
          <Button variant={"link"} className="text-base text-left pt-0 pb-0 pl-0 pr-0 -mr-1 w-full h-auto justify-start">
            <p className="text-sm whitespace-nowrap font-normal overflow-hidden text-ellipsis">{title}</p>
          </Button>
        </Link>
        <div className='w-[40px]'>
          <div className="bg-foreground min-w-0 h-5 text-background text-xs text-center rounded-full px-1">
            <span className="text-xs h-5 flex flex-col justify-center">
            {(documentCount <= 999)?documentCount.toString():"999+"}
            </span>
          </div>
        </div>
			</div>
		</div>
	);
}