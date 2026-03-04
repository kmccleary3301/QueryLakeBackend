"use client";

import { useContextAction } from "@/app/context-provider";
import { usePathname } from "next/navigation";
import Sidebar from "./sidebar-app";
import { useEffect, useState } from "react";
import DocSidebar from "./sidebar-docs";
import ApiSidebar from "./sidebar-api";


type sidebar_assigned = "app" | "api" | "documentation" | "none"

export default function SidebarController() {
	const {
    authReviewed,
    loginValid,
  } = useContextAction();

	const pathname = usePathname();
	const [sidebarAssignment, setSidebarAssignment] = useState<sidebar_assigned>("none");

	useEffect(() => {
		// if (!authReviewed || !loginValid) {
		// 	setSidebarAssignment("none");
		// 	return;
		// }
		// console.log("Pathname changed to", pathname, pathname?.startsWith("/app"), pathname?.startsWith("/nodes"));
		if (pathname?.startsWith("/app") || 
			pathname?.startsWith("/nodes") || 
			pathname?.startsWith("/themes") ||
			pathname?.startsWith("/collection") ||
			pathname?.startsWith("/organizations")
		) {
			// console.log("Setting sidebar assignment to app");
			setSidebarAssignment("app");
		} else if (pathname?.startsWith("/docs")) {
			setSidebarAssignment("documentation");
		} else if (pathname?.startsWith("/platform")) {
			setSidebarAssignment("api");
		} else {
			setSidebarAssignment("none");
		}
	}, [pathname, authReviewed, loginValid])

	// useEffect(() => {
	// 	console.log("Sidebar: sidebarAssignment changed to", sidebarAssignment);
	// }, [sidebarAssignment])


	// switch(sidebarAssignment) {
	// 	case "app":
	// 		return <Sidebar/>
	// 	case "api":
	// 		return null
	// 	case "documentation":
	// 		return <DocSidebar/>
	// 	default:
	// 		return null;
	// }

	return (
		<>
			{(pathname?.startsWith("/app") || 
				pathname?.startsWith("/nodes") || 
				pathname?.startsWith("/themes") ||
				pathname?.startsWith("/collection") ||
				pathname?.startsWith("/settings") ||
				pathname?.startsWith("/organizations")
			) && authReviewed && loginValid && 
				<Sidebar/>
			}
			{(pathname?.startsWith("/docs")) && 
				<DocSidebar/>
			}
      {(pathname?.startsWith("/platform")) && authReviewed && loginValid && 
				<ApiSidebar/>
			}
			{/* {sidebarAssignment === "api" && <DocSidebar/>} */}
		</>
	)
}