"use client";
import React, { useRef, useState, useEffect } from "react";
import { uploadVideo, processVideoOnServer, streamDashboard } from "../lib/api";

export default function VideoPlayer({ fileInputId, onDetections }: { fileInputId?: string; onDetections?: any }) {
  const [job, setJob] = useState(null);
  const [logs, setLogs] = useState<string[]>([]);
  useEffect(() => {
    const unsub = streamDashboard((data) => {
      setLogs((s) => [...s, JSON.stringify(data)]);
      // Optionally call onDetections with server payload
      onDetections?.([data]);
    });
    return () => unsub();
  }, [onDetections]);

  async function handleUploadAndProcess() {
    const input = document.getElementById(fileInputId || "video-input") as HTMLInputElement | null;
    const f = input?.files?.[0];
    if (!f) return alert("Choose a video first");
    const up = await uploadVideo(f);
    const res = await processVideoOnServer(up.path, { card_conf: 0.5, frame_skip: 5 });
    setJob(res.summary ?? res);
    setLogs((s) => [...s, "Server summary: " + JSON.stringify(res)]);
  }

  return (
    <div>
      <div className="mb-3">
        <button onClick={handleUploadAndProcess} className="px-4 py-2 bg-indigo-600 text-white rounded">Upload & Process</button>
      </div>
      <div className="text-xs text-slate-500 space-y-1">
        {logs.map((l, i) => <div key={i}>{l}</div>)}
      </div>
      {job && <pre className="mt-2 text-xs">{JSON.stringify(job, null, 2)}</pre>}
    </div>
  );
}