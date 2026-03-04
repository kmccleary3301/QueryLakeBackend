'use client'

// ATTEMPT 1
import { motion } from "framer-motion";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useContextAction } from "@/app/context-provider";
import { useRouter } from 'next/navigation';
import { userDataType } from "@/types/globalTypes";
import { getCookie } from "@/hooks/cookies";



export function Template({ 
  children 
}: {
  children: React.ReactNode;
}) {

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ type: "spring", ease: "linear", bounce: 0, duration: 2.5, delay: 0.2}}
    >
      {children}
    </motion.div>
  );
}

export default function RootTemplate({ 
    children 
}: {
    children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);


  const { 
    authReviewed,
    loginValid,
    userData,
    getUserData
  } = useContextAction();

  const onMount = useCallback(async() => {
    if (userData === undefined) {
			const cookie : string | undefined = await getCookie({ key: 'UD', convert_object : false });
			getUserData(cookie, () => {setMounted(true);});
		}
  }, [getUserData, userData]);

  useEffect(() => {
    onMount();
  }, [onMount]);

  useEffect(() => {
    // console.log("Calling effect with userdata and mount change")
    if (mounted && !authReviewed) {
      getUserData(userData?.auth, () => {setMounted(true);});
    }
  }, [userData?.auth, mounted, authReviewed, getUserData]);
  
  useEffect(() => {
    // console.log("pathname changed to:", pathname);
  }, [pathname]);
  
  // Redirect to login page if not logged in, redirect to home page if logged in and attempting to log in.
  useEffect(() => {
    // console.log("TEMPLATE RECIEVED VALUES:", authReviewed, loginValid, pathname, pathname?.startsWith("/account"));
    if (!authReviewed) return;
    if (pathname?.startsWith("/docs")) return; // Docs can be viewed regardless of login status.

    // console.log("TEMPLATE 2");
    if (!loginValid && !(pathname?.startsWith("/auth"))) {
      console.log("Redirecting to login page");
      router.push("/auth/login");
    } else if (loginValid && (pathname?.startsWith("/auth"))) {
      console.log("Redirecting to workspace selection");
      router.push("/select-workspace");
    }
  }, [authReviewed, loginValid, pathname, mounted, router]);

  return (
    <>
      {(authReviewed) ? (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ type: "spring", ease: "linear", bounce: 0, duration: 2.5, delay: 0.2}}
        >
          {children}
        </motion.div>
      ) : ( // TODO: Add loading spinner
        <div className="w-full h-full">

        </div>
      )}
    </>
  );
}

// ATTEMPT 2

// import { motion, AnimatePresence } from "framer-motion";

// export default function Template({ 
//     children
// }: {
//     children: React.ReactNode;
// }) {
//   return (
//     // <AnimatePresence mode="wait">
//       <motion.div
//         initial={{ opacity: 0 }}
//         animate={{ opacity: 1 }}
//         exit={{ opacity: 0 }}
//         transition={{ type: "spring", ease: "linear", bounce: 0, duration: 5}}
//       >
//         {children}
//       </motion.div>
//     // </AnimatePresence  >
//   );
// }

// ATTEMPT 3

// import React from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import { usePathname } from "next/navigation";
// import { LayoutRouterContext } from "next/dist/shared/lib/app-router-context.shared-runtime";
// import { useContext, useRef } from "react";

// // Prevents instant page opening
// function FrozenRouter(props: { children: React.ReactNode }) {
//   const context = useContext(LayoutRouterContext ?? {});
//   const frozen = useRef(context).current;

//   return (
//     <LayoutRouterContext.Provider value={frozen}>
//       {props.children}
//     </LayoutRouterContext.Provider>
//   );
// }


// export default function Template({children}: {children: React.ReactNode}){
    
//   let pathname = usePathname();

//   return(
//     <>
//       <AnimatePresence exitBeforeEnter>
//         <motion.div
//           key={pathname} 
//           initial={{opacity: 0}}
//           animate={{opacity: 1}}
//           exit={{opacity: 0, transition: { duration: 5}}}
//           transition={{ type: "spring", ease: "linear", bounce: 0, duration: 5}}
//         >
//           <FrozenRouter>{children}</FrozenRouter>
//         </motion.div>    
//       </AnimatePresence> 
//     </>
//   )
// }


// ATTEMPT 4

// "use client";

// import { AnimatePresence, motion } from "framer-motion";
// import { useRouter } from "next/navigation";
// import {
//   createContext,
//   MouseEventHandler,
//   PropsWithChildren,
//   use,
//   useTransition,
// } from "react";

// export const DELAY = 200;

// const sleep = (ms: number) =>
//   new Promise<void>((resolve) => setTimeout(() => resolve(), ms));
// const noop = () => {};

// type TransitionContext = {
//   pending: boolean;
//   navigate: (url: string) => void;
// };
// const Context = createContext<TransitionContext>({
//   pending: false,
//   navigate: noop,
// });
// export const usePageTransition = () => use(Context);
// export const usePageTransitionHandler = () => {
//   const { navigate } = usePageTransition();
//   const onClick: MouseEventHandler<HTMLAnchorElement> = (e) => {
//     e.preventDefault();
//     const href = e.currentTarget.getAttribute("href");
//     if (href) navigate(href);
//   };

//   return onClick;
// };

// type Props = PropsWithChildren<{
//   className?: string;
// }>;

// export default function Template({ children, className }: Props) {
//   const [pending, start] = useTransition();
//   const router = useRouter();
//   const navigate = (href: string) => {
//     start(async () => {
//       router.push(href);
//       await sleep(DELAY);
//     });
//   };

//   const onClick: MouseEventHandler<HTMLDivElement> = (e) => {
//     const a = (e.target as Element).closest("a");
//     if (a) {
//       e.preventDefault();
//       const href = a.getAttribute("href");
//       if (href) navigate(href);
//     }
//   };

//   return (
//     <Context.Provider value={{ pending, navigate }}>
//       <div onClickCapture={onClick} className={className}>
//         {children}
//       </div>
//     </Context.Provider>
//   );
// }

// export function Animate({ children, className }: Props) {
//   const { pending } = usePageTransition();
//   return (
//     <AnimatePresence>
//       {!pending && (
//         <motion.div
//           initial={{ opacity: 0 }}
//           animate={{ opacity: 1 }}
//           exit={{ opacity: 0 }}
//           className={className}
//         >
//           {children}
//         </motion.div>
//       )}
//     </AnimatePresence>
//   );
// }
