import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "./hover-card"

export function HoverTextDiv({
  className,
  hint,
  children,
}:{
  className?: string,
  hint: string,
  children: React.ReactNode
}) {
  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <div className={className}>
          {children}
        </div>
      </HoverCardTrigger>
      <HoverCardContent className="w-auto max-w-[200px]">
        <p className="text-sm">
          {hint}
        </p>
      </HoverCardContent>
    </HoverCard>
  )
}
