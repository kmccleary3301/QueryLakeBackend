const next = require('next')
const express = require('express');
// const { proxy } = require('http-proxy-middleware');
const { createProxyMiddleware } = require('http-proxy-middleware');

const port = parseInt(process.env.PORT, 10) || 3000;
const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

app.prepare().then(() => {
    const server = express();

    server.use(
        '/api/ping',
        createProxyMiddleware({
            target: "http://localhost:8000/api/ping",
            changeOrigin: true,
        }),
    );
 
    server.all('*', (req, res) => handle(req, res));

    server.listen(port, err => {
        if (err) throw err;
        console.log(`> Ready on http://localhost:${port}`);
    });
});


// const { createServer } = require('http')
// const { parse } = require('url')
// const next = require('next')
// const express = require('express');
// const proxy = require('http-proxy-middleware');
 
// const dev = process.env.NODE_ENV !== 'production'
// const hostname = 'localhost'
// const port = 3000
// // when using middleware `hostname` and `port` must be provided below
// const app = next({ dev, hostname, port })
// const handle = app.getRequestHandler()
 
// console.log("SERVER STARTED")
// app.prepare().then(() => {
//   createServer(async (req, res) => {
//     try {
//       // Be sure to pass `true` as the second argument to `url.parse`.
//       // This tells it to parse the query portion of the URL.
//       const parsedUrl = parse(req.url, true)
//       const { pathname, query } = parsedUrl
 
//       if (pathname === '/a') {
//         await app.render(req, res, '/a', query)
//       } else if (pathname === '/b') {
//         await app.render(req, res, '/b', query)
//       } else {
//         await handle(req, res, parsedUrl)
//       }
//     } catch (err) {
//       console.error('Error occurred handling', req.url, err)
//       res.statusCode = 500
//       res.end('internal server error')
//     }
//   })
//     .once('error', (err) => {
//       console.error(err)
//       process.exit(1)
//     })
//     .listen(port, () => {
//       console.log(`> Ready on http://${hostname}:${port}`)
//     })
// })