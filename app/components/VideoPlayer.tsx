"use client";

import React, { useEffect, useRef, useState } from "react";
import { uploadVideo, processVideoOnServer, streamDashboard } from "../../lib/api";

export default function VideoPlayer({ fileInputId, onDetections }: { fileInputId?: string; onDetections?: (data:any)=>void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [job, setJob] = useState<any>(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    const unsub = streamDashboard((data) => {
      setLogs((s) => [...s, "SSE: " + JSON.stringify(data)]);
      onDetections?.([data]);
    });
    return () => unsub();
  }, [onDetections]);

  useEffect(() => {
    return () => { if (fileUrl) URL.revokeObjectURL(fileUrl); };
  }, [fileUrl]);

  function handleLocalSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFileUrl(URL.createObjectURL(f));
    setLogs((s) => [...s, `Selected local: ${f.name}`]);
  }

  async function handleUploadAndProcess() {
    const input = fileInputId ? document.getElementById(fileInputId) as HTMLInputElement | null : document.getElementById("video-input") as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (!file) return alert("Choose a video first");
    setProcessing(true);
    setLogs((s) => [...s, `Uploading ${file.name}...`]);
    try {
      const up = await uploadVideo(file);
      setLogs((s) => [...s, `Uploaded: ${up.filename}`]);
      // server returns a filesystem path; processing endpoint consumes that path
      const res = await processVideoOnServer(up.path, { card_conf: 0.5, frame_skip: 5 });
      setLogs((s) => [...s, `Server summary: ${JSON.stringify(res)}`]);
      setJob(res.summary ?? res);
      onDetections?.(res.summary ? [res.summary] : []);
    } catch (err) {
      console.error(err);
      setLogs((s) => [...s, "Error uploading/processing"]);
    } finally {
      setProcessing(false);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-3 items-center">
        {/* file input: uses provided id if given so page's input can be used */}
        {!fileInputId && (
          <input id="video-input" type="file" accept="video/*" onChange={handleLocalSelect} className="block" />
        )}
        <button onClick={handleUploadAndProcess} className="px-4 py-2 bg-indigo-600 text-white rounded" disabled={processing}>
          {processing ? "Uploading..." : "Upload & Process"}
        </button>
        <div className="text-sm text-slate-500">{job ? "Job ready" : processing ? "Processing" : "Idle"}</div>
      </div>

      <div className="bg-slate-900/5 rounded-md overflow-hidden">
        <video ref={videoRef} src={fileUrl || undefined} controls className="w-full max-h-64 bg-black" />
      </div>

      <div className="text-xs text-slate-500 space-y-1 max-h-40 overflow-auto">
        {logs.map((l, i) => <div key={i}>{l}</div>)}
      </div>

      {job && (
        <pre className="mt-2 text-xs bg-white/5 p-2 rounded">{JSON.stringify(job, null, 2)}</pre>
      )}
    </div>
  );
}