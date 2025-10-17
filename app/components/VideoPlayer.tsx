"use client";

import React, { useRef, useState, useEffect } from "react";

interface Props {
  fileInputId?: string;
  onDetections?: (demoList: Array<any>) => void;
}

export default function VideoPlayer({ fileInputId, onDetections }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [processing, setProcessing] = useState<boolean>(false);
  const [playing, setPlaying] = useState<boolean>(false);
  const [stopped, setStopped] = useState<boolean>(false);
  const [fileUrl, setFileUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  useEffect(() => {
    const input = fileInputId ? document.getElementById(fileInputId) as HTMLInputElement | null : null;
    if (!input) return;
    const inputEl = input as HTMLInputElement;
    function onChange() {
      const files = inputEl.files;
      const f = files && files[0];
      if (f) {
        if (fileUrl) URL.revokeObjectURL(fileUrl);
        const url = URL.createObjectURL(f);
        setFileUrl(url);
        setProcessing(false);
        setPlaying(false);
        setStopped(false);
      }
    }
    inputEl.addEventListener("change", onChange);
    return () => inputEl.removeEventListener("change", onChange);
  }, [fileInputId, fileUrl]);

  function handleProcess() {
    const v = videoRef.current;
    setProcessing(true);
    setStopped(false);
    if (v) {
      v.play().then(() => setPlaying(true)).catch(() => {});
    }
    setTimeout(() => {
      const demoList = [
        { id: "v-rec-1", name: "Person A", thumb: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=200&q=60&auto=format&fit=crop", status: "recognized" },
        { id: "v-un-1", name: "Unknown #1", thumb: "https://images.unsplash.com/photo-1549719389-5e63d49a2c8f?w=200&q=60&auto=format&fit=crop", status: "unrecognized" },
        { id: "v-rec-2", name: "Person B", thumb: "https://images.unsplash.com/photo-1545996124-1b1d9d0e6e8d?w=200&q=60&auto=format&fit=crop", status: "recognized" },
      ];
      onDetections?.(demoList);
    }, 1200);
  }

  function handlePauseResume() {
    const v = videoRef.current;
    if (!v) return;
    if (playing) {
      v.pause();
      setPlaying(false);
      setStopped(false);
    } else {
      v.play().then(() => setPlaying(true)).catch(() => {});
    }
  }

  function handleStopResume() {
    const v = videoRef.current;
    if (!v) return;
    if (!stopped) {
      v.pause();
      setPlaying(false);
      setStopped(true);
      setProcessing(false);
    } else {
      v.play().then(() => {
        setPlaying(true);
        setStopped(false);
        setProcessing(true);
      }).catch(() => {});
    }
  }

  return (
    <div className="space-y-3">
      <div className="bg-slate-900/5 rounded-lg overflow-hidden border">
        <video ref={videoRef} src={fileUrl || "/footballVideo.mp4"} className="w-full max-h-64 bg-black" controls={false} />
      </div>

      <div className="flex items-center gap-3">
        <button onClick={handleProcess} className="px-4 py-2 bg-indigo-600 text-white rounded-lg shadow">
          {processing ? "Processing (▶)" : "Process Video"}
        </button>

        <button onClick={handlePauseResume} disabled={!processing && !playing && !stopped} className="px-3 py-2 rounded-lg border bg-white text-sm">
          {playing ? "Pause" : "Resume"}
        </button>

        <button onClick={handleStopResume} disabled={!processing && !playing && !stopped} className="px-3 py-2 rounded-lg border bg-white text-sm">
          {stopped ? "Resume" : "Stop"}
        </button>

        <div className="text-sm text-slate-500 ml-auto">{stopped ? "Stopped" : playing ? "Playing" : processing ? "Processing" : "Idle"}</div>
      </div>
    </div>
  );
}