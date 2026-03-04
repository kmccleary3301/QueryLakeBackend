"use client";

import Flow from './Flow';

export default function FlowDisplay() {
  return (
    <div className='h-full w-full'>
			<div className="h-full w-full flex flex-col bg-background">
				<Flow />
			</div>
    </div>
  );
}