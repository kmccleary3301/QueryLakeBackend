// import {
//   View,
//   Text,
// } from 'react-native';
// import { ScrollView, TextInput } from 'react-native-gesture-handler';
// import { Feather } from '@expo/vector-icons';
// import { useEffect, useState } from 'react';
// import CollectionWrapper from './CollectionWrapper';
import CollectionWrapper from '../manual_components/collection-wrapper';
import { collectionGroup, selectedCollectionsType, userDataType } from '@/types/globalTypes';
import { Button } from '@/components/ui/button';
import * as Icon from 'react-feather';
import { Input } from '@/components/ui/input';
import Link from 'next/link';

type SidebarCollectionSelectProps = {
  onChangeCollections?: (collectionGroups: collectionGroup[]) => void,
  userData: userDataType,
  collectionGroups: collectionGroup[],
  setCollectionGroups: React.Dispatch<React.SetStateAction<collectionGroup[]>>,
  setCollectionSelected: (collection_hash_id : string, value : boolean) => void,
  selectedCollections: selectedCollectionsType,
  scrollClassName: string,
}

export default function SidebarCollectionSelect(props: SidebarCollectionSelectProps) {

  const toggleMyCollections = (selected: boolean, group_key: number) => {
		// if (selected) {
		for (let i = 0; i < props.collectionGroups[group_key].collections.length; i++) {
			props.collectionGroups[group_key].toggleSelections?.[i].setSelected(selected);
		}
  };

  return (
    <div className='h-full w-full flex flex-col'>
      <div className="w-full pt-0 pb-[10px]">
        <div className="flex flex-row bg-gray-800 rounded-2xl">
            <Input
              className="text-sm h-8 outline-none"
              spellCheck={false}
              placeholder={'Search Public Collections'}
            />
        </div>
      </div>
      <div className="w-full pl-0 pr-0 flex flex-col">
        <div>
          <div className={props.scrollClassName}>
            <div className='space-y-3'>
              {props.collectionGroups.map((v, k) => (
                <CollectionWrapper
                  key={k}
                  title={v.title}
                  onToggleSelected={(selected: boolean) => {
                    toggleMyCollections(selected, k);
                    if (props.onChangeCollections) { props.onChangeCollections(props.collectionGroups); }
                  }}
                  collections={v.collections}
                  setCollectionSelected={props.setCollectionSelected}
                  collectionSelected={props.selectedCollections}
                />
              ))}
              {/* <div className='h-[800px]'/> */}
            </div>
            <div className='pt-[10px] pb-[20px]'>
              <Button className="flex w-full rounded-full h-10 items-center justify-center" variant={"ghost"}>
                <Link href={"/collection/create"} className='flex items-center'>
                  <Icon.Plus size={20}/>
                  <p className="text-base ml-2 pt-[1px]">New Collection</p>
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div> 
  );
}