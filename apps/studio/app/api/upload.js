// File: pages/api/upload.js

import fetch from 'node-fetch';

export default async function handler(req, res) {
  const backendUrl = 'http://localhost:8000/upload_document';

  if (req.method === 'POST') {
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: req.body,
        headers: req.headers,
      });

      const data = await response.json();

      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: 'Error proxying request' });
    }
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}