const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export async function uploadVideo(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/api/upload/video`, { method: "POST", body: fd });
  return res.json();
}

export async function processVideoOnServer(path: string, opts?: { card_conf?: number; frame_skip?: number }) {
  const fd = new FormData();
  fd.append("path", path);
  if (opts?.card_conf !== undefined) fd.append("card_conf", String(opts.card_conf));
  if (opts?.frame_skip !== undefined) fd.append("frame_skip", String(opts.frame_skip));
  const res = await fetch(`${BASE}/api/process/video`, { method: "POST", body: fd });
  return res.json();
}

export async function processImagesOnServer(files: File[]) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const res = await fetch(`${BASE}/api/process/images`, { method: "POST", body: fd });
  return res.json();
}

export async function uploadYolo(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/api/load-yolo`, { method: "POST", body: fd });
  return res.json();
}

export function streamDashboard(onMessage: (data: any) => void) {
  const es = new EventSource(`${BASE}/api/dashboard/stream`);
  es.onmessage = (ev) => {
    try { onMessage(JSON.parse(ev.data)); } catch (e) { console.warn(e); }
  };
  es.onerror = (e) => { console.error("SSE error", e); es.close(); };
  return () => es.close();
}