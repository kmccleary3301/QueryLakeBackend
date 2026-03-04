import { useEffect, useState, ReactNode, useCallback } from 'react';
import { selectedCollectionsType } from '@/types/globalTypes';
import { Button } from '@/components/ui/button';
import { motion, useAnimation } from "framer-motion";
import * as Icon from 'react-feather';
import CollectionPreview from '@/components/manual_components/collection-preview';
import { redirect } from 'next/navigation'
// import { fontSans } from '@/lib/fonts';

type collectionType = {
  title: string,
  items: number,
  hash_id: string,
  type: string,
}

type CollectionWrapperProps = {
	onToggleCollapse?: (opened: boolean) => void,
	onToggleSelected?: (selected: boolean) => void,
	children?: ReactNode,
	title: string,
  collections: collectionType[],
  setCollectionSelected: (collection_hash_id : string, value : boolean) => void,
  collectionSelected: selectedCollectionsType
}

export default function CollectionWrapper(props: CollectionWrapperProps) {
	const [opened, setOpened] = useState(false);
	
  
	const controlHeight = useAnimation();
	const selectionCircleSize = useAnimation();

  const countSelected = useCallback(() => {
    const mapped_selected = props.collections.map((value) => props.collectionSelected.get(value.hash_id)).filter(Boolean).length;
    return mapped_selected;
  }, [props.collectionSelected, props.collections]);

  const [selectedCount, setSelectedCount] = useState<number>(countSelected());
  const [selected, setSelected] = useState(() => (selectedCount === props.collections.length));
  const [mixedSelection, setMixedSelection] = useState(true);
  
	useEffect(() => {
		controlHeight.set({
			height: 0
		});
	}, [controlHeight]);

	useEffect(() => {
		controlHeight.start({
			height: (opened && props.collections.length > 0)?"auto":0,
			transition: { duration: 0.4 }
		});
  }, [opened, props.collections, controlHeight]);

  useEffect(() => {
		selectionCircleSize.start({
			width: selected?11:0,
			height: selected?11:0,

			transition: { duration: 0.1 }
		});
  }, [selected, selectionCircleSize]);

  useEffect(() => {
    // console.log("SELECTED COUNT:", selectedCount);
    if (selectedCount === props.collections.length) {
      setSelected(true);
    }
  }, [props.collectionSelected, selectedCount, props.collections.length]);

	// useEffect(() => {
	// 	console.log("PROP COLLECTIONS:", props.collections);
	// }, [props.collections]);

  return (
    <>
    {(props.collections.length === 0)?(
      <></>
    ):(

    

    <div className={"flex flex-col px-3 bg-secondary rounded-md w-full"}>
      <div className="h-11 flex flex-row w-full space-x-2">
        <div className='w-5 h-auto flex flex-col justify-center'> 
        <Button variant={"default"} className="w-5 h-5 rounded-full bg-theme-one flex items-center justify-center p-0" onClick={() => {
          setMixedSelection(false);
          for (let i = 0; i < props.collections.length; i++) {
            props.setCollectionSelected(props.collections[i].hash_id, !selected);
          }
          setSelectedCount((prevCount) => (selected)?0:props.collections.length);
          setSelected(selected => !selected);
        }}>
          <motion.div animate={selectionCircleSize} initial={{
            width: selected?11:0,
            height: selected?11:0,
          }} className="rounded-full bg-background"/>
        </Button>
        </div>
        <div className="flex-grow flex flex-col justify-center h-auto">

          <p className="text-sm whitespace-nowrap overflow-hidden text-ellipsis">{props.title}</p>
        </div>
        <div className='h-auto flex flex-col justify-center'>
          <Button className="h-7 w-7 p-0 rounded-full hover:bg-primary/20 active:bg-primary/10" variant={"ghost"}
            onClick={() => {
              if (props.onToggleCollapse) { props.onToggleCollapse(!opened); }
              setOpened(opened => !opened);
            }}
          >
            <Icon.ChevronDown
              style={{
                transform: opened?"rotate(0deg)":"rotate(90deg)"
              }}
            />
          </Button>
        </div>
      </div>
      <motion.div animate={controlHeight} className="text-sm antialiased w-full overflow-hidden">
        <div className="overflow-hidden space-y-3 pb-2 pt-2 pr-1">
          {props.collections.map((value : collectionType, index: number) => (
            <CollectionPreview
              key={index}
              title={value.title} 
              documentCount={value.items}
              collectionId={value.hash_id}
              onToggleSelected={(collection_selected: boolean) => {
                props.setCollectionSelected(value.hash_id, collection_selected);
                setSelectedCount((prevCount) => prevCount + ((collection_selected)?1:-1));
                
                if (selected && !collection_selected) {
                  // selected_values[index][1](false);
                  setMixedSelection(true);
                  setSelected(false);
                }
                // if (props.onChangeCollections) { props.onChangeCollections(CollectionGroups); }
              }}
              parentSelected={selected}
              parentMixedSelection={mixedSelection}
              selectedPrior={props.collectionSelected.get(value.hash_id)}
            />
          ))}
        </div>
      </motion.div>
    </div>
    )}
    </>
  );
}
