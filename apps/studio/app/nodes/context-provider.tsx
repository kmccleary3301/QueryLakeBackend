// ThemeProvider.tsx
'use client';
import {
  Dispatch,
	PropsWithChildren,
	SetStateAction,
	createContext,
	useContext,
	useCallback,
	useEffect,
	useRef,
  useState,
} from 'react';
import { displaySection } from '@/types/toolchain-interface';
import { useNodesState, useEdgesState, Edge, Node, NodeChange, EdgeChange, ReactFlowInstance } from 'reactflow';
import CustomNode, { ToolchainNodeData, ToolchainNodeReactFlow } from './node_editor/components/CustomNode';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { fetchToolchainConfig } from '@/hooks/querylakeAPI';
import { useContextAction } from '../context-provider';
import { ToolChain, createAction, feedMapping, sequenceAction, staticRoute } from '@/types/toolchains';

type OnChange<ChangesType> = (changes: ChangesType[]) => void;

type nodeSearchParamArgs = {
  toolchainId?: string,
  referenceId?: string,
  mode: "create" | "edit",
}

type routeType = (string | number | {route: (string | number)[]})[];

const get_feed_route = (route : Array<sequenceAction>) => {
  // let static_route : (string | number)[] = [];
  for (let i = 0; i < route.length - 1; i++) {
    if (typeof route[i] === "object") {
      continue;
    }
    return route[i] as string;
  }
  if (typeof route[route.length - 1] === "object") {
    const current_obj : createAction = route[route.length - 1] as createAction;
    if (current_obj.route && current_obj.route.length > 0) {
      return current_obj.route[0];
    }
  }
  return undefined;
}

const initNodes = [
  {
    id: '1',
    type: 'custom',
    data: { name: 'Jane Doe', job: 'CEO', emoji: '😎' },
    position: { x: 0, y: 50 },
  },
  {
    id: '2',
    type: 'custom',
    data: { name: 'Tyler Weary', job: 'Designer', emoji: '🤓' },
    position: { x: -200, y: 200 },
  },
  {
    id: '3',
    type: 'custom',
    data: { name: 'Kristi Price', job: 'Developer', emoji: '🤩' },
    position: { x: 200, y: 200 },
  },
];

const initEdges = [
  {
    id: 'e1-2',
    source: '1',
    target: '2',
  },
  {
    id: 'e1-3',
    source: '1',
    target: '3',
  },
];


const NodeContext = createContext<{
	interfaceConfiguration: displaySection;
	setInterfaceConfiguration: (value: displaySection) => void;
	getInterfaceConfiguration: () => displaySection;
  toolchainNodes: Node<object, string | undefined>[];
  setToolchainNodes: Dispatch<SetStateAction<Node<object, string | undefined>[]>>;
  onNodesChange: OnChange<NodeChange>;
  toolchainEdges: Edge<any>[];
  setToolchainEdges: Dispatch<SetStateAction<Edge<any>[]>>;
  onEdgesChange: OnChange<EdgeChange>;
  reactFlowInstance: ReactFlowInstance<any, any> | null;
  setReactFlowInstance: Dispatch<SetStateAction<ReactFlowInstance<any, any> | null>>;
}>({
	interfaceConfiguration: {
		split: "none",
		size: 100,
		align: "center",
		tailwind: "",
		mappings: []
	},
	setInterfaceConfiguration: () => {
		return {
			split: "none",
			size: 100,
			align: "center",
			tailwind: "",
			mappings: []
		}
	},
	getInterfaceConfiguration: () => {
		return {
			split: "none",
			size: 100,
			align: "center",
			tailwind: "",
			mappings: []
		}
	},
  toolchainNodes: [],
  setToolchainNodes: () => [],
  onNodesChange: () => [],
  toolchainEdges: [],
  setToolchainEdges: () => [],
  onEdgesChange: () => [],
  reactFlowInstance: null,
  setReactFlowInstance: () => {},
});

export const NodeContextProvider = ({
	interfaceConfiguration,
	children,
}: PropsWithChildren<{ 
	interfaceConfiguration : displaySection,
}>) => {

  const {
    userData,
  } = useContextAction();

  const router = useRouter(),
        pathname = usePathname(),
        search_params = useSearchParams();


  const m_param = search_params?.get("mode") || "create";
  const initial_mode = ((["create", "edit"].indexOf(m_param) > -1)?m_param:"create") as "create" | "edit";
  const [searchArgs, setSearchArgs] = useState<nodeSearchParamArgs>({
    ...(search_params?.get("t_id"))?{toolchainId: search_params?.get("t_id") as string}:{},
    ...(search_params?.get("ref"))?{referenceId: search_params?.get("ref") as string}:{},
    mode: "create",
  });
  const [nodes, set_nodes, on_nodes_change] = useNodesState<object>(initNodes);
  const [edges, set_edges, on_edges_change] = useEdgesState(initEdges);
  const [react_flow_instance, set_react_flow_instance] = useState<ReactFlowInstance<any, any> | null>(null);
  const [referenceToolchain, setReferenceToolchain] = useState<ToolChain | null>(null);
  const referenceToolchainID = useRef<string | null>(null);

	const interface_configuration = useRef<displaySection>(interfaceConfiguration);
	const set_interface_configuration = (value: displaySection) => {
    // console.log("Setting interface configuration", value);
		interface_configuration.current = value;
	};
	const get_interface_configuration : () => displaySection = () => {
		return interface_configuration.current;
	};

  const loadInToolchain = useCallback((toolchain: ToolChain) => {
    console.log("Loading in Toolchain", toolchain);
    setReferenceToolchain(toolchain);

    let toolchain_nodes : ToolchainNodeData[] = [];
    let toolchain_edges : Edge<any>[] = [];
    let iterator = 0;

    if (toolchain.display_configuration) {
      set_interface_configuration(toolchain.display_configuration);
    } else {
      set_interface_configuration({
        split: "none",
        size: 100,
        align: "center",
        tailwind: "",
        mappings: []
      });
    }

    for (let i = 0; i < toolchain.nodes.length; i++) {
      toolchain_nodes.push({
        id: toolchain.nodes[i].id,
        type: "toolchain",
        data: toolchain.nodes[i],
        position: {x: i*200, y: 200},
      });
      
      for (let j = 0; j < (toolchain.nodes[i].feed_mappings || []).length; j++) {
        const feed : feedMapping = (toolchain.nodes[i].feed_mappings || [])[j];
        const feed_route = get_feed_route(feed.sequence || []) || "||UNCLASSIFIED||";

        if (feed.destination === "<<STATE>>" || feed.destination === "<<USER>>") {
          continue;
        }

        console.log("Feed", feed);
        console.log(feed_route);

        toolchain_edges.push({
          type: 'turbo',
          id: `e${i}-${j}`,
          source: toolchain.nodes[i].id,
          sourceHandle: `feed-${j}`,
          target: feed.destination,
          targetHandle: feed_route as string,
        });
      }
    }

    set_nodes((prevNodes) => [
      // ...prevNodes, 
      ...toolchain_nodes
    ]);

    console.log("Setting nodes", toolchain_nodes);
    console.log("Setting edges", toolchain_edges);

    set_edges((prevEdges) => [
      // ...prevEdges, 
      ...toolchain_edges
    ]);
  }, [set_nodes, set_edges]);

  useEffect(() => {
    const m_param = search_params?.get("mode") || "create";
    const initial_mode = ((["create", "edit"].indexOf(m_param) > -1)?m_param:"create") as "create" | "edit";
    setSearchArgs({
      ...(search_params?.get("t_id"))?{toolchainId: search_params?.get("t_id") as string}:{},
      ...(search_params?.get("ref"))?{referenceId: search_params?.get("ref") as string}:{},
      mode: initial_mode,
    });
  }, [search_params]);

  useEffect(() => {
    let get_id : string | null = null;

    console.log("Search args hook called with args", searchArgs, referenceToolchainID.current);

    if (searchArgs.mode === "create" && 
        searchArgs.referenceId) {
      get_id = searchArgs.referenceId;
    } else if (searchArgs.mode === "edit" && 
               searchArgs.toolchainId) {
      get_id = searchArgs.toolchainId;
    } else if (searchArgs.mode === "create") {
      get_id = null;
      referenceToolchainID.current = get_id;
      set_edges([]);
      set_nodes([]);
      return;
    }

    if (get_id && (get_id !== referenceToolchainID.current || referenceToolchainID.current === null)) {
      referenceToolchainID.current = get_id;
      fetchToolchainConfig({
        auth: userData?.auth as string,
        toolchain_id: get_id as string,
        onFinish: (data : ToolChain) => {
          loadInToolchain(data);
        }
      })
    }
  }, [searchArgs, loadInToolchain, set_edges, set_nodes, userData?.auth]);

	return (
		<NodeContext.Provider value={{ 
			interfaceConfiguration: interface_configuration.current,
			setInterfaceConfiguration: set_interface_configuration,
			getInterfaceConfiguration: get_interface_configuration,
      toolchainNodes: nodes,
      setToolchainNodes: set_nodes,
      onNodesChange: on_nodes_change,
      toolchainEdges: edges,
      setToolchainEdges: set_edges,
      onEdgesChange: on_edges_change,
      reactFlowInstance: react_flow_instance,
      setReactFlowInstance: set_react_flow_instance,
		}}>
			{children}
		</NodeContext.Provider>
	);
};

export const useNodeContextAction = () => {
	return useContext(NodeContext);
};
