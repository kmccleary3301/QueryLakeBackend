"use client";
import {
  Breadcrumb,
  BreadcrumbEllipsis,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuTrigger,
	DropdownMenuSubTrigger,
	DropdownMenuSubContent,
	DropdownMenuPortal
} from "@/components/ui/dropdown-menu"
import { Fragment } from "react"
import RouteAddSheet from "./route-add-sheet"

export function RouteEntryMenu({
	onDelete,
	onModify,
	sub = false,
	routeElement,
	children,
} : {
	onDelete: () => void,
	onModify: (value: string | number) => void,
	sub?: boolean,
	routeElement: (string | number),
	children: React.ReactNode
}) {

	return (
		<>
		{sub ? (
			<DropdownMenuSub>
				<DropdownMenuSubTrigger className="flex items-center gap-1">
					{children}
				</DropdownMenuSubTrigger>
				<DropdownMenuPortal>
					<DropdownMenuSubContent>
						<DropdownMenuItem onClick={onDelete}>Delete</DropdownMenuItem>
						{/* <DropdownMenuItem onClick={onModify}>Modify</DropdownMenuItem> */}
						{/* <RouteAddSheet 
							value={routeElement} 
							type={(typeof routeElement === "string")?"string":"number"}
							onChange={(value) => {onModify(value)}}
						>
							<DropdownMenuItem>Modify</DropdownMenuItem>
						</RouteAddSheet> */}
					</DropdownMenuSubContent>
				</DropdownMenuPortal>
			</DropdownMenuSub>
		) : (
			<DropdownMenu>
				<DropdownMenuTrigger className="flex items-center gap-1">
					{children}
				</DropdownMenuTrigger>
				<DropdownMenuContent align="start">
					<DropdownMenuItem onClick={onDelete}>Delete</DropdownMenuItem>
					{/* <RouteAddSheet 
						value={routeElement} 
						type={(typeof routeElement === "string")?"string":"number"}
						onChange={(value) => {onModify(value)}}
					>
						<DropdownMenuItem>Modify</DropdownMenuItem>
					</RouteAddSheet> */}
				</DropdownMenuContent>
			</DropdownMenu>
		)}
		</>
	)

}

export default function RouteSetter({
	onRouteSet,
	route
}:{
	onRouteSet: (route: (string | number)[]) => void,
	route: (string | number)[]
}) {
	return (
		<Breadcrumb>
			<BreadcrumbList>
				<BreadcrumbSeparator/>
				{(route.length <= 3) ? (
					<>
						{route.map((r, index) => (
							<Fragment key={index}>
								<BreadcrumbItem>
									<RouteEntryMenu
										routeElement={r}
										onDelete={() => {onRouteSet([...route.slice(0, index), ...route.slice(index+1)])}}
										onModify={() => {console.log("Modify (Not Implemented")}}
									>
										<span className="text-primary">{r}</span>
									</RouteEntryMenu>
								</BreadcrumbItem>
								{index < route.length-1 && <BreadcrumbSeparator/>}
							</Fragment>
						))}
					</>
				) : (
					<>
						{route.slice(0, 1).map((r, index) => (
							<BreadcrumbItem key={index}>
								<RouteEntryMenu
									routeElement={r}
									onDelete={() => {onRouteSet([...route.slice(0, index), ...route.slice(index+1)])}}
									onModify={() => {console.log("Modify (Not Implemented")}}
								>
									<span className="text-primary">{r}</span>
								</RouteEntryMenu>
							</BreadcrumbItem>
						))}
						<BreadcrumbSeparator/>
						<BreadcrumbItem>
							<DropdownMenu>
								<DropdownMenuTrigger className="flex items-center gap-1">
									<BreadcrumbEllipsis className="h-4 w-4" />
									<span className="sr-only">Toggle menu</span>
								</DropdownMenuTrigger>
								<DropdownMenuContent align="start">
									{route.slice(1, route.length-1).map((r, index) => (
											<RouteEntryMenu
												key={index}
												routeElement={r}
												onDelete={() => {onRouteSet([...route.slice(0, index+1), ...route.slice(index+2)])}}
												onModify={() => {console.log("Modify (Not Implemented")}}
												sub={true}
											>
												<span className="text-primary">{r}</span>
											</RouteEntryMenu>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
						</BreadcrumbItem>
						<BreadcrumbSeparator/>
						{route.slice(route.length-1).map((r, index) => (
							<Fragment key={index}>
								<BreadcrumbItem>
									<RouteEntryMenu
										routeElement={r}
										onDelete={() => {onRouteSet([...route.slice(0, index+route.length-1)])}}
										onModify={() => {console.log("Modify (Not Implemented")}}
									>
										<span className="text-primary">{r}</span>
									</RouteEntryMenu>
								</BreadcrumbItem>

								{/* {index < 1 && <BreadcrumbSeparator/>} */}
							</Fragment>
						))}
					</>
				)}
			</BreadcrumbList>
		</Breadcrumb>
	)
}