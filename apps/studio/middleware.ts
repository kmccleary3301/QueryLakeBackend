import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(_request: NextRequest) {
  if (process.env.NODE_ENV === "development") {
    return NextResponse.next();
  }
  return new NextResponse("Not found", { status: 404 });
}

export const config = {
  matcher: ["/test/:path*", "/sink/:path*", "/all_pages_panel/:path*"],
};

