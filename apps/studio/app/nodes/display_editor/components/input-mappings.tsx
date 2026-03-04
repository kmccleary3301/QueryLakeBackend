"use client";
import { Skeleton } from "@/components/ui/skeleton";
import { displayMapping } from "@/types/toolchain-interface";
import { Input } from "@/components/ui/input";
import { 
	Trash2,
	Plus
} from 'lucide-react';
import { Button } from "@/components/ui/button";
import RouteSetter from "./route-setter";
import RouteAddSheet from "./route-add-sheet";

export function DisplayComponentSkeletonMapper({
	info
}:{
	info: displayMapping,
}) {
	switch(info.display_as) {
		case "chat":
			return (
				<div className="flex flex-row space-x-4 w-auto	">
					<div>
						<Skeleton className="rounded-full w-10 h-10"/>
					</div>
					<div className="flex-grow flex flex-col space-y-3">
						{Array(10).fill(0).map((_, i) => (
							<Skeleton key={i} className="rounded-full w-auto h-3"/>
						))}
					</div>
				</div>
			);
		case "markdown":
			return (
				<div>
					<h2>Markdown</h2>
				</div>
			);
		default:
			return (
				<div>
					<h2>Unknown</h2>
				</div>
			);
	}
}

export default function DisplayMappings({
	info,
	onDelete,
	onRouteSet,
}:{
	info: displayMapping,
	onDelete: () => void,
	onRouteSet: (route: (string | number)[]) => void
}) {

	return (
		<div className="flex flex-col space-y-4 w-auto">
			<div className="flex flex-row space-x-2 min-h-12 pt-2">
				<div className="flex-grow h-auto flex flex-col justify-center">
					<RouteSetter onRouteSet={onRouteSet} route={info.display_route}/>
				</div>
				<div className="pr-2">
					<RouteAddSheet
						type={"string"}
						onChange={(value) => {
							onRouteSet([...info.display_route, value]);
						}}
					>
						<Button size="icon" variant="ghost">
								<Plus className="w-4 h-4 text-primary" />
						</Button>
					</RouteAddSheet>
				</div>
				<Button size="icon" variant="ghost" onClick={onDelete}>
          <Trash2 className="w-4 h-4 text-primary" />
        </Button>
			</div>
			<DisplayComponentSkeletonMapper info={info}/>
		</div>
	)
}