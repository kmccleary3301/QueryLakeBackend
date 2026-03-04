export default function DisplayLayout({ 
	children 
} : {
	children: React.ReactNode
}) {

	return (
		<div 
			className="w-full h-[calc(100vh)] absolute" 
			style={{
				"--container-width": "100%"
			} as React.CSSProperties}
		>
			{children}
		</div>
	)
}