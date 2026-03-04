"use client";
import { HoverTextDiv } from "@/components/ui/hover-text-div";
import { indexRouteRetrievedInputArgValue, indexRouteRetrievedNew, indexRouteRetrievedOutputArgValue, indexRouteRetrievedStateValue, staticRoute, staticRouteElementType } from "@/types/toolchains";

export const staticRouteElementTypes = [
  "string",
  "int",
  "stateValue",
  "nodeInput",
  "nodeOutput",
];

type staticRouteElementTypes = "string" | "int" | "stateValue" | "nodeInput" | "nodeOutput";


export function StaticRouteElementCreation({
  value,
  className = "",
}:{
  value: staticRouteElementType,
  className?: string,
}) {
  let type = value;
  
  if (typeof value === "string") { // String case
    return (
      <input type="number" className={className}/>
    );
  } else if (typeof value === "number") { // Number case
    return (
      <input type="number" className={className}/>
    );
  } else if (typeof value === "object") { // Object case
    if ((value as indexRouteRetrievedStateValue).getFromState !== undefined) {
      return (
        <div className="flex flex-row">
          <HoverTextDiv hint="Value from State" className="h-auto flex flex-col justify-center">
            <p className="text-xs w-5 h-5 rounded-full border-2 border-[#fed734]">S</p>
          </HoverTextDiv>
          <StaticRouteCreation values={(value as indexRouteRetrievedStateValue).getFromState.route} className={className}/>
        </div>
      )
    } else if ((value as indexRouteRetrievedInputArgValue).getFromInputs !== undefined) {
      return (
        <div className="flex flex-row">
          <HoverTextDiv hint="Node Input" className="h-auto flex flex-col justify-center">
            <p className="text-xs w-5 h-5 rounded-full border-2 border-[#2a8af6]">I</p>
          </HoverTextDiv>
          <StaticRouteCreation values={(value as indexRouteRetrievedInputArgValue).getFromInputs.route} className={className}/>
        </div>
      )
    } else if ((value as indexRouteRetrievedOutputArgValue).getFromOutputs !== undefined) {
      return (
        <div className="flex flex-row">
          <HoverTextDiv hint="Node Output" className="h-auto flex flex-col justify-center">
            <p className="text-xs w-5 h-5 rounded-full border-2 border-[#e92a67]">O</p>
          </HoverTextDiv>
          <StaticRouteCreation values={(value as indexRouteRetrievedOutputArgValue).getFromOutputs.route} className={className}/>
        </div>
      )
    }
  }
  return (
    <h1>Error</h1>
  )
}

export function StaticRouteCreation({
  values,
  disabled = false,
  className = "",
}:{
  values : staticRoute,
  disabled?: boolean,
  className?: string,
}) {

  return (
    <div className={className}>
      {values.map((value : staticRouteElementType, index : number) => (
        <div key={index} className="flex flex-row">
          <StaticRouteElementCreation value={value} className="ml-2"/>
        </div>
      ))}
    </div>
  )
}