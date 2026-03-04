"use client";
import React, { useCallback, useMemo, useRef, useState } from 'react';
import ReactFlow, { useNodesState, useEdgesState, addEdge, MiniMap, Controls, Connection, Edge, Background, ReactFlowInstance, ReactFlowProvider } from 'reactflow';

import 'reactflow/dist/base.css';

import CustomNode, { ToolchainNodeReactFlow } from './CustomNode';
import ContextMenuWrapper from './context-menu-wrapper';
import { useNodeContextAction } from "../../context-provider";
import { useThemeContextAction } from '@/app/theme-provider';

import 'reactflow/dist/base.css';
import './turbo_style.css';
import TurboNode, { TurboNodeData } from './TurboNode';
import TurboEdge from './TurboEdge';
import ToolchainNode from './ToolchainNode';
import { hslStringToRGBHex } from '@/hooks/rgb-hsl-functions';

// const nodeTypes = {
//   custom: CustomNode,
//   toolchainNode: ToolchainNodeReactFlow
// };

export default function Flow() {
  let id = 50;
  const getId = () => `${id++}`;

  const { 
    toolchainNodes,
    setToolchainNodes,
    onNodesChange,
    toolchainEdges,
    setToolchainEdges,
    onEdgesChange,
    reactFlowInstance,
    setReactFlowInstance,
  } = useNodeContextAction();

  const {
    themeBrightness
  } = useThemeContextAction();

  // const [nodes, setNodes, onNodesChange] = useNodesState<object>(initNodes);
  // const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges);
  const reactFlowWrapper = useRef(null);

  const onConnect = useCallback(
    (params: Connection | Edge) => setToolchainEdges((eds) => addEdge(params, eds)),
    [setToolchainEdges]
  );

  const nodeTypes = useMemo(
    () => ({
      custom: CustomNode,
      // toolchainNode: ToolchainNodeReactFlow,
      toolchain: ToolchainNode,
      turbo: TurboNode,
    }),
    [],
  );

  const edgeTypes = useMemo(
    () => ({
      turbo: TurboEdge,
    }),
    [],
  );

  const defaultEdgeOptions = useMemo(
    () => ({
      type: 'turbo',
      markerEnd: 'edge-circle',
    }),
    [],
  );

  return (
    <div className="flex-grow text-xs">
      <ContextMenuWrapper reactFlowInstance={reactFlowInstance} setNodes={setToolchainNodes} getId={getId}>
        <ReactFlowProvider>
          <div className="reactflow-wrapper w-full h-full" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={toolchainNodes}
            edges={toolchainEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            onInit={setReactFlowInstance}
          >
            <MiniMap color="#F00" className='bg-background' zoomable pannable/>
            <Controls />
            <Background 
              color={hslStringToRGBHex(themeBrightness.foreground) as string} 
              style={{backgroundColor: hslStringToRGBHex(themeBrightness.background) as string}}
              gap={16}
            />
            {/* <Controls showInteractive={false} /> */}
            <svg>
              <defs>
                <linearGradient id="edge-gradient">
                  <stop offset="0%" stopColor="#ae53ba" />
                  <stop offset="100%" stopColor="#2a8af6" />
                </linearGradient>

                <marker
                  id="edge-circle"
                  viewBox="-5 -5 10 10"
                  refX="0"
                  refY="0"
                  markerUnits="strokeWidth"
                  markerWidth="10"
                  markerHeight="10"
                  orient="auto"
                >
                  <circle stroke="#2a8af6" strokeOpacity="0.75" r="2" cx="0" cy="0" />
                </marker>
              </defs>
            </svg>
          </ReactFlow>
            {/* <ReactFlow
              nodes={toolchainNodes}
              edges={toolchainEdges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              defaultEdgeOptions={defaultEdgeOptions}
              
              fitView
              onInit={setReactFlowInstance}
              // className="bg-teal-50"
              className=""
            >
              <MiniMap zoomable pannable/>
              <Controls />
              <Background color="#aaa" gap={16}/>
            </ReactFlow> */}
          </div>
        </ReactFlowProvider>
      </ContextMenuWrapper>
    </div>
  );
};

// export default Flow;
