/** @type {import('next').NextConfig} */
const nextConfig = {
  // headers: async() => {
  //   return [
  //     {
  //       // matching all API routes  
  //       source: "/api/:path*",
  //       headers: [
  //         { key: "Access-Control-Allow-Credentials", value: "true" },
  //         { key: "Access-Control-Allow-Origin", value: "*" }, // replace this your actual origin
  //         { key: "Access-Control-Allow-Methods", value: "GET,DELETE,PATCH,POST,PUT" },
  //         { key: "Access-Control-Allow-Headers", value: "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version" },
  //       ]
  //     }
  //   ]
  // },
  rewrites: async() => {
		return [
			// {
			// 	source: '/api/:path*',
			// 	destination: `http://localhost:8000/api/:path*`,
			// }
      {
        source: "/upload/:path*",
        destination: "http://localhost:8000/upload_document/:path*"
      },
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*"
      },
      {
        source: "/v2/kernel/:path*",
        destination: "http://localhost:8000/v2/kernel/:path*"
      },
      {
        source: "/sessions/:path*",
        destination: "http://localhost:8000/sessions/:path*"
      },
      {
        source: "/toolchain",
        destination: "http://localhost:8000/toolchain"
      }
		]
	},
  reactStrictMode: true,
  serverExternalPackages: ['shiki', 'vscode-oniguruma'],
  // NOTE: We run `npm run lint` and `npm run typecheck` explicitly in CI/build
  // to avoid Next's build-time lint/typecheck worker OOM/SIGKILL behavior.
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // swcMinify: true,
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "avatars.githubusercontent.com",
      },
      {
        protocol: "https",
        hostname: "images.unsplash.com",
      },
    ],
  },
  redirects() {
    return [
      {
        source: "/components",
        destination: "/docs/components/accordion",
        permanent: true,
      },
      {
        source: "/examples",
        destination: "/examples/mail",
        permanent: false,
      },
      {
        source: "/aceternity",
        destination: "/aceternity/mail",
        permanent: false,
      },
      {
        source: "/nodes",
        destination: "/nodes/node_editor",
        permanent: false,
      },
      {
        source: "/auth",
        destination: "/auth/login",
        permanent: false,
      },
      {
        source: "/docs",
        destination: "/docs/naive_bayes_classifier",
        permanent: true,
      },
      {
        source: "/app",
        destination: "/app/create",
        permanent: true,
      }
    ]
  },
}

export default nextConfig;
