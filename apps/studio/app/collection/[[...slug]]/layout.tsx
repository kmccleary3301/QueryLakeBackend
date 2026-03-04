import { ReactQueryProvider } from "./react-query";

export default function Layout({
		children,
	}: Readonly<{
		children: React.ReactNode;
	}>) {

		return (
			<ReactQueryProvider>
				{children}
			</ReactQueryProvider>
		)
	}