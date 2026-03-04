"use client";
import React, { memo, ReactNode } from 'react';
import { Handle, NodeProps, Position } from 'reactflow';
// import { FiCloud } from 'react-icons/fi';

export type TurboNodeData = {
  title: string;
  icon?: ReactNode;
  subline?: string;
};

const TurboNode = memo(function TurboNode({ data }: NodeProps<TurboNodeData>) {
  return (
    <>
      <div className="cloud gradient">
        <div>
          {/* <FiCloud /> */}
        </div>
      </div>
      <Handle type="target" position={Position.Left} className='w-5 h-5 z-10 rounded-full border-2 border-[#e92a67]'>
        <div className="h-full flex flex-col justify-center">
          <div className='w-2 h-2 rounded-full bg-black m-auto'/>
        </div>
      </Handle>

      <Handle type="source" position={Position.Right} className='w-5 h-5 z-10 rounded-full border-2 border-[#e92a67]'>
        {/* <div className="cloud gradient pointer-events-none">
          <div>
          </div>
        </div> */}
      </Handle>
      <div className="wrapper gradient">
        <div className="inner">
          <div className="body">
            {data.icon && <div className="icon">{data.icon}</div>}
            <div>
              <div className="title">{data.title}</div>
              {data.subline && <div className="subline">{data.subline}</div>}
            </div>
          </div>
          
          {/* <Handle type="source" position={Position.Right} /> */}
        </div>
      </div>
    </>
  );
});

TurboNode.displayName = "TurboNode";

export default TurboNode;
