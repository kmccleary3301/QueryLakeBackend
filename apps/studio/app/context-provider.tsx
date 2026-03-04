// ThemeProvider.tsx
'use client';
import {
	Dispatch,
	PropsWithChildren,
	SetStateAction,
	createContext,
	useCallback,
	useContext,
	useEffect,
	useState,
} from 'react';
import { 
	selectedCollectionsType, 
	userDataType,
	setStateOrCallback,
	toolchain_session,
	collectionGroup,
  APIFunctionSpec
} from '@/types/globalTypes';
import { deleteCookie, getCookie, setCookie } from '@/hooks/cookies';
import craftUrl from '@/hooks/craftUrl';
import { QuerylakeFunctionHelp, fetchToolchainConfig, fetchToolchainSessions, getUserCollections } from '@/hooks/querylakeAPI';
import { ToolChain } from '@/types/toolchains';
import { BundledTheme } from 'shiki/themes';

export type breakpointType = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export type codeThemePreferenceType = {theme: BundledTheme, backgroundColor?: string, textColor?: string};

const Context = createContext<{
	userData: userDataType | undefined;
	setUserData: Dispatch<SetStateAction<userDataType | undefined>>;

	collectionGroups: collectionGroup[];
	setCollectionGroups: Dispatch<SetStateAction<collectionGroup[]>>;
	refreshCollectionGroups: () => void;

	selectedCollections: selectedCollectionsType;
	setSelectedCollections: Dispatch<SetStateAction<selectedCollectionsType>>;

	toolchainSessions : Map<string, toolchain_session>,
	setToolchainSessions : React.Dispatch<React.SetStateAction<Map<string, toolchain_session>>>,
  refreshToolchainSessions : () => void,

	activeToolchainSession : string | undefined,
	setActiveToolchainSession : setStateOrCallback<string | undefined>,

	selectedToolchain : string | undefined,
	setSelectedToolchain : setStateOrCallback<string | undefined>,

  selectedToolchainFull : ToolChain | undefined,
	setSelectedToolchainFull : setStateOrCallback<ToolChain>,

	authReviewed : boolean,
	setAuthReviewed : setStateOrCallback<boolean>,

	loginValid : boolean,
	setLoginValid : setStateOrCallback<boolean>,

  shikiTheme : codeThemePreferenceType,
  setShikiTheme : setStateOrCallback<codeThemePreferenceType>,

	getUserData : (user_data_input : string | undefined, onFinish : () => void) => void,

  apiFunctionSpecs : APIFunctionSpec[] | undefined,

	breakpoint : breakpointType,

	sidebarOpen : boolean,
	setSidebarOpen : setStateOrCallback<boolean>,
}>({
	userData: undefined,
	setUserData: () => undefined,

	collectionGroups: [],
	setCollectionGroups: () => [],
	refreshCollectionGroups: () => {},

	selectedCollections: new Map(),
	setSelectedCollections: () => new Map(),

	toolchainSessions: new Map(),
	setToolchainSessions: () => new Map(),
  refreshToolchainSessions: () => {},

	activeToolchainSession: undefined,
	setActiveToolchainSession: () => undefined,

	selectedToolchain: undefined,
	setSelectedToolchain: () => undefined,

  selectedToolchainFull: undefined,
  setSelectedToolchainFull: () => undefined,

	authReviewed: false,
	setAuthReviewed: () => false,

	loginValid: false,
	setLoginValid: () => false,

  shikiTheme: {theme: 'tokyo-night', backgroundColor: undefined},
  setShikiTheme: () => {return {theme: 'tokyo-night', backgroundColor: undefined}},

	getUserData: () => undefined,

  apiFunctionSpecs: undefined,

	breakpoint: '2xl',

	sidebarOpen: true,
	setSidebarOpen: () => true,
});


type login_results = {
  success: false,
  error: string
} | {
  success: true,
  result: userDataType
};


export const ContextProvider = ({
	userData,
	selectedCollections,
	toolchainSessions,
	children,
}: PropsWithChildren<{ 
	userData : userDataType | undefined , 
	selectedCollections : selectedCollectionsType,
	toolchainSessions : Map<string, toolchain_session>
}>) => {

	// const [user_data, set_user_data] = useState<userDataType | undefined>(getCookie({ key: 'UD' }) as userDataType | undefined);
	const [user_data, set_user_data] = useState<userDataType | undefined>(userData);
	const [collection_groups, set_collection_groups] = useState<collectionGroup[]>([]);
	const [selected_collections, set_selected_collections] = useState<selectedCollectionsType>(selectedCollections);
	const [toolchain_sessions, set_toolchain_sessions] = useState<Map<string, toolchain_session>>(toolchainSessions);
	const [active_toolchain_session, set_active_toolchain_session] = useState<string | undefined>(undefined);
	const [selected_toolchain, set_selected_toolchain] = useState<string | undefined>(undefined);

  const [selected_toolchain_full, set_selected_toolchain_full] = useState<ToolChain | undefined>(undefined);

	const [auth_reviewed, set_auth_reviewed] = useState<boolean>(false);
	const [login_valid, set_login_valid] = useState<boolean>(false);
	const [break_point, set_breakpoint] = useState<breakpointType>('2xl');
  const [shiki_theme, set_shiki_theme] = useState<codeThemePreferenceType>({theme: 'tokyo-night', backgroundColor: undefined});
  const [api_function_specs, set_api_function_specs] = useState<APIFunctionSpec[] | undefined>(undefined);
	const [sidebar_open, set_sidebar_open] = useState<boolean>(true);


	useEffect(() => {
    const breakpoints : breakpointType[] = ['xs', 'sm', 'md', 'lg', 'xl', '2xl'];
    const widths = [0, 640, 768, 1024, 1280, 1536];

		const updateBreakpoint = () => {
			const width = window.innerWidth;
			let index = widths.findIndex(w => width < w);
			if (index !== -1) {
				index = index === 0 ? 0 : index - 1;
			} else {
				index = breakpoints.length - 1;
			}
			set_breakpoint(breakpoints[index]);
		};
    window.addEventListener('resize', updateBreakpoint);
    updateBreakpoint();
    return () => window.removeEventListener('resize', updateBreakpoint);
  }, []);

	// useEffect(() => {console.log("BREAKPOINT:", break_point)}, [break_point]);

	const get_user_data = async (user_data_input : string | undefined, onFinish : () => void) => {
		
		const cookie_ud = await getCookie({ key: 'UD', convert_object: false }) as string | undefined;
		console.log("Cookie UD:", cookie_ud);
		
		if (user_data_input !== undefined) {
			fetch(`/api/login`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					auth: user_data_input,
				}),
			}).then((response) => {
				response.json().then(async (data : login_results) => {
					console.log("GOT LOGIN DATA:", data);
					if (data.success) {
						await setCookie({ key: "UD", value: data.result.auth });
						
						set_user_data(data.result);

						set_login_valid(true);
						set_auth_reviewed(true);

						onFinish();
					} else {
						await deleteCookie({ key: 'UD' });
						set_user_data(undefined);

						set_login_valid(false);
						set_auth_reviewed(true);

						onFinish();
					}
				});
			});
		} else {
			set_login_valid(false);
			set_auth_reviewed(true);

			onFinish();
		}
	};
  

  const refresh_toolchain_sessions = useCallback(() => {
		console.log("Auth Updated:", user_data);
    fetchToolchainSessions({
      auth: user_data?.auth as string,
      onFinish: (v : toolchain_session[]) => {
        const newToolchainSessions = new Map<string, toolchain_session>();
        v.forEach((session : toolchain_session) => {
          newToolchainSessions.set(session.id, session);
        });
        set_toolchain_sessions(newToolchainSessions);
      }
    })
  }, [user_data]);

	const refresh_collection_groups = () => {
		getUserCollections({
			auth: user_data?.auth as string, 
			set_value: set_collection_groups
		});
	};

  const setFullToolchain = useCallback((toolchain_id : string) => {
    if (!auth_reviewed || !user_data?.auth) return;
    // console.log("toolchain_id:", toolchain_id);
    fetchToolchainConfig({
      auth: user_data?.auth as string,
      toolchain_id: toolchain_id as string,
      onFinish: (v : ToolChain) => set_selected_toolchain_full(v)
    })
    refresh_toolchain_sessions();
  }, [user_data?.auth, auth_reviewed, refresh_toolchain_sessions]);

  const getAPIFunctionSpecs = useCallback(() => {
    QuerylakeFunctionHelp({
      // auth: user_data?.auth as string,
      onFinish: (v : APIFunctionSpec[] | false) => {
        if (v) {
          set_api_function_specs(v);
        }
      }
    })
  }, []);


  useEffect(() => {
    if (selected_toolchain === undefined) return;
    setFullToolchain(selected_toolchain);
  }, [selected_toolchain, user_data?.auth, auth_reviewed, setFullToolchain]);
  
  useEffect(() => {
    if (user_data === undefined || !auth_reviewed) return;
    set_selected_toolchain(user_data.default_toolchain.id);
    getAPIFunctionSpecs();
  }, [user_data, auth_reviewed, getAPIFunctionSpecs]);
	
	return (
		<Context.Provider value={{ 
			userData : user_data, 
			setUserData : set_user_data,
			collectionGroups : collection_groups,
			setCollectionGroups : set_collection_groups,
			refreshCollectionGroups : refresh_collection_groups,
			selectedCollections : selected_collections,
			setSelectedCollections : set_selected_collections,
			toolchainSessions : toolchain_sessions,
			setToolchainSessions : set_toolchain_sessions,  
      refreshToolchainSessions : refresh_toolchain_sessions,
			activeToolchainSession : active_toolchain_session,
			setActiveToolchainSession : set_active_toolchain_session,
			selectedToolchain : selected_toolchain,
			setSelectedToolchain : set_selected_toolchain,
      selectedToolchainFull : selected_toolchain_full,
      setSelectedToolchainFull : set_selected_toolchain_full,
			authReviewed : auth_reviewed,
			setAuthReviewed : set_auth_reviewed,
			loginValid : login_valid,
			setLoginValid : set_login_valid,
      shikiTheme : shiki_theme,
      setShikiTheme : set_shiki_theme,
			getUserData : get_user_data,
      apiFunctionSpecs: api_function_specs,
			breakpoint : break_point,
			sidebarOpen : sidebar_open,
			setSidebarOpen : set_sidebar_open,
		}}>
			{children}
		</Context.Provider>
	);
};

export const useContextAction = () => {
	return useContext(Context);
};
