"use client";

import React from "react";
import ResultsTabs from "./ResultsTabs";
import VideoPlayer from "./VideoPlayer";
import { uploadYolo } from "../../lib/api";

type PersonCard = { name: string; status: "Active" | "Idle" | string; img: string };
type SampleStats = {
  detections: number;
  emotions: Record<string, number>;
  cards: { red: number; yellow: number; green: number };
};

interface Props {
  active: string;
  recognized: any[];
  unrecognized: any[];
  sampleStats: SampleStats;
  handleFilePlaceholder: (e: React.ChangeEvent<HTMLInputElement> | null, label: string) => void;
  processImages: (files?: FileList | null) => void;
  processVideoDetections: (demoList?: any[]) => void;
}

export default function MainPanel({
  active,
  recognized,
  unrecognized,
  sampleStats,
  handleFilePlaceholder,
  processImages,
  processVideoDetections,
}: Props) {
  const demoPeople: PersonCard[] = [
    {
      name: "Alex Turner",
      status: "Active",
      img: "https://images.unsplash.com/photo-1545996124-1b1d9d0e6e8d?w=400&q=60&auto=format&fit=crop",
    },
    {
      name: "Priya Singh",
      status: "Idle",
      img: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&q=60&auto=format&fit=crop",
    },
    {
      name: "Liam Johnson",
      status: "Active",
      img: "https://images.unsplash.com/photo-1549719389-5e63d49a2c8f?w=400&q=60&auto=format&fit=crop",
    },
  ];

  return (
    <div className="rounded-2xl h-auto bg-white p-6 shadow-md">
      <div className="flex items-start justify-between gap-6">
        {active !== "Home" && (
          <div>
            <h2 className="text-xl font-semibold">{active}</h2>
            <p className="text-sm text-slate-500 mt-1">
              Interactive panel (frontend-only). Pick files or use the controls below.
            </p>
          </div>
        )}

        <div className="flex flex-col items-end gap-4 w-full">
          {active === "Home" ? (
            <div className="hidden md:block w-full">
              <div className="relative overflow-hidden rounded-2xl shadow-2xl bg-black w-full h-64 sm:h-72 md:h-80 lg:h-96 transition-all">
                <video
                  autoPlay
                  loop
                  muted
                  playsInline
                  disablePictureInPicture
                  controls={false}
                  src="/footballVideo.mp4"
                  className="absolute inset-0 w-full h-full object-cover select-none rounded-2xl z-0"
                >
                  <source src="/footballVideo.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>

                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent rounded-2xl pointer-events-none z-10" />

                <div className="absolute bottom-5 left-6 text-white z-20">
                  <h3 className="text-lg font-semibold tracking-wide drop-shadow">AI in Motion</h3>
                  <p className="text-sm text-gray-300">Experience the power of intelligence</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="text-sm text-slate-500">Status</div>
              <div className="rounded-full bg-emerald-100 text-emerald-700 px-3 py-1 text-sm">Ready</div>
            </div>
          )}
        </div>
      </div>

      {/* Image tab */}
      {active === "Image" && (
        <div className="mt-6 space-y-4">
          <label className="block text-sm font-medium text-slate-700">Select Images</label>
          <input id="image-input" type="file" accept="image/*" multiple onChange={(e) => handleFilePlaceholder(e, "Images")} className="block w-full text-sm" />
          <div className="flex items-center gap-3 mt-4">
            <button onClick={() => processImages((document.getElementById("image-input") as HTMLInputElement | null)?.files)} className="px-4 py-2 bg-indigo-600 text-white rounded-lg shadow">
              Process Images
            </button>
          </div>
          <div className="mt-4">
            <ResultsTabs recognized={recognized} unrecognized={unrecognized} emptyMessage="No image results yet" />
          </div>
        </div>
      )}

      {/* Video tab */}
      {active === "Video" && (
        <div className="mt-6 space-y-4">
          <label className="block text-sm font-medium text-slate-700">Upload / Select Video</label>
          <input id="video-input" type="file" accept="video/*" onChange={(e) => handleFilePlaceholder(e, "Video")} className="block w-full text-sm" />
          <div className="mt-4">
            <VideoPlayer fileInputId="video-input" onDetections={processVideoDetections} />
          </div>
          <div className="mt-4">
            <ResultsTabs recognized={recognized} unrecognized={unrecognized} emptyMessage="No video detections yet" />
          </div>
        </div>
      )}

      {/* Training / Model / Dashboard preserved */}
      {active === "Training" && (
        <div className="mt-6 space-y-4">
          <label className="block text-sm font-medium text-slate-700">Upload Training Archive / Images</label>
          <input
            type="file"
            {...({ webkitdirectory: "true", directory: "true" } as any)}
            multiple
            onChange={(e) => handleFilePlaceholder(e, "Training")}
            className="block w-full text-sm text-slate-600"
          />
          <div className="flex items-center gap-3 mt-4">
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg shadow hover:brightness-95">Start Training (placeholder)</button>
            <button onClick={() => {}} className="px-4 py-2 bg-indigo-600 text-white rounded-lg shadow hover:brightness-95">Append</button>
          </div>
        </div>
      )}

      {active === "Model Viewer" && (
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border p-4">
              <h3 className="text-sm font-medium">Known Models</h3>
              <p className="mt-2 text-sm text-slate-500">No backend connected — list will appear after integration.</p>
            </div>
            <div className="rounded-lg border p-4">
              <h3 className="text-sm font-medium">YOLO Model</h3>
              <p className="mt-2 text-sm text-slate-500">Upload .pt to preview (placeholder)</p>
              <input
                type="file"
                accept=".pt"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  const res = await uploadYolo(f);
                  if (res?.ok) {
                    // show a message or keep model path
                    alert("YOLO model uploaded (server): " + (res.path ?? f.name));
                  } else {
                    alert("YOLO upload failed");
                  }
                }}
                className="mt-3 text-sm"
              />
            </div>
          </div>
        </div>
      )}

      {active === "Dashboard" && (
        <div className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-white shadow">
              <div className="text-sm text-slate-500">Detections</div>
              <div className="text-2xl font-semibold">{sampleStats.detections}</div>
            </div>

            <div className="p-4 rounded-xl bg-white shadow">
              <div className="text-sm text-slate-500">Emotions (Top 3)</div>
              <div className="mt-3 space-y-2">
                {Object.entries(sampleStats.emotions).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between">
                    <div className="capitalize">{k}</div>
                    <div className="font-medium">{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-white shadow">
              <div className="text-sm text-slate-500">Cards</div>
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between"><div>Red</div><div className="font-medium">{sampleStats.cards.red}</div></div>
                <div className="flex items-center justify-between"><div>Yellow</div><div className="font-medium">{sampleStats.cards.yellow}</div></div>
                <div className="flex items-center justify-between"><div>Green</div><div className="font-medium">{sampleStats.cards.green}</div></div>
              </div>
            </div>
          </div>

          <div className="md:col-span-3 grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
            {demoPeople.map((p) => (
              <div key={p.name} className="flex items-center gap-4 bg-white rounded-xl p-4 shadow">
                <img src={p.img} alt={p.name} className="w-16 h-16 rounded-lg object-cover" />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{p.name}</div>
                      <div className="text-xs text-slate-500">Demo person</div>
                    </div>
                    <div className={`text-xs font-medium px-2 py-1 rounded-full ${p.status === "Active" ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                      {p.status}
                    </div>
                  </div>
                  <div className="mt-2 text-sm text-slate-500">Quick preview: 2 detections • best score 0.82</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}