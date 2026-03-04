"use client";
import { Button } from "@/components/ui/button"


export default function PingButton() {
	return (
		<Button onClick={() => {
			console.log("Hi!");
			fetch("/api/python").then((response) => {response.text().then((data : string) => {
				console.log("Got data:", data);
			});})
		}}>Ping</Button>
	);
}