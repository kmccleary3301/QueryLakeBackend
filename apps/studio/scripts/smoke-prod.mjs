import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const PORT = process.env.PORT ?? "37201";
const BASE_URL = `http://127.0.0.1:${PORT}`;

const checks = [
  { path: "/", expect: 200 },
  { path: "/select-workspace", expect: 200 },
  { path: "/status", expect: 200 },
  { path: "/w/test-workspace/dashboard", expect: 200 },
  { path: "/w/test-workspace/settings", expect: 200 },
  { path: "/definitely-not-a-route", expect: 404 },
  { path: "/test", expect: 404 },
  { path: "/sink", expect: 404 },
  { path: "/all_pages_panel", expect: 404 },
];

function startServer() {
  const child = spawn("npx", ["next", "start", "-p", PORT], {
    stdio: "inherit",
    env: { ...process.env, NODE_ENV: "production" },
  });
  return child;
}

async function waitForServer() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`${BASE_URL}/`, { redirect: "manual" });
      if (res.status > 0) return;
    } catch {
      // ignore
    }
    await delay(500);
  }
  throw new Error(`Timed out waiting for server at ${BASE_URL}`);
}

async function main() {
  const child = startServer();

  const cleanup = () => {
    if (!child.killed) {
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 1500).unref();
    }
  };
  process.on("exit", cleanup);
  process.on("SIGINT", () => process.exit(130));
  process.on("SIGTERM", () => process.exit(143));

  try {
    await waitForServer();

    const failures = [];
    for (const check of checks) {
      const url = `${BASE_URL}${check.path}`;
      const res = await fetch(url, { redirect: "manual" });
      const ok = res.status === check.expect;
      const label = ok ? "OK" : "FAIL";
      // eslint-disable-next-line no-console
      console.log(`${label} ${check.path} -> ${res.status} (expected ${check.expect})`);
      if (!ok) failures.push({ ...check, got: res.status });
    }

    if (failures.length > 0) {
      throw new Error(
        `Smoke failed: ${failures
          .map((f) => `${f.path}=${f.got} (expected ${f.expect})`)
          .join(", ")}`
      );
    }
  } finally {
    cleanup();
    await delay(250);
  }
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});

