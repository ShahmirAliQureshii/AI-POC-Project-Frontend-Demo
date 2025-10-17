'use client'
import { useState } from "react";

export default function ResultsTabs({ recognized = [], unrecognized = [], emptyMessage = "No results" }) {
  const [tab, setTab] = useState("recognized");

  return (
    <div className="bg-white rounded-xl border p-3 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setTab("recognized")}
            className={`px-3 py-1 rounded-lg text-sm font-medium ${tab === "recognized" ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-100"}`}
          >
            Recognized ({recognized.length})
          </button>
          <button
            onClick={() => setTab("unrecognized")}
            className={`px-3 py-1 rounded-lg text-sm font-medium ${tab === "unrecognized" ? "bg-rose-500 text-white" : "text-slate-600 hover:bg-slate-100"}`}
          >
            Unrecognized ({unrecognized.length})
          </button>
        </div>
        <div className="text-xs text-slate-400">Results (demo)</div>
      </div>

      <div className="mt-3">
        {tab === "recognized" ? (
          recognized.length ? (
            <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {recognized.map((r) => (
                <li key={r.id} className="flex items-center gap-3 bg-slate-50 p-2 rounded-md">
                  <img src={r.thumb} alt={r.name} className="w-12 h-12 rounded-md object-cover" />
                  <div className="flex-1">
                    <div className="font-medium text-sm">{r.name}</div>
                    <div className="text-xs text-slate-500">Matched • demo</div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-sm text-slate-500">{emptyMessage}</div>
          )
        ) : unrecognized.length ? (
          <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {unrecognized.map((r) => (
              <li key={r.id} className="flex items-center gap-3 bg-slate-50 p-2 rounded-md">
                <img src={r.thumb} alt={r.name} className="w-12 h-12 rounded-md object-cover" />
                <div className="flex-1">
                  <div className="font-medium text-sm">{r.name}</div>
                  <div className="text-xs text-slate-500">Unknown • demo</div>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm text-slate-500">{emptyMessage}</div>
        )}
      </div>
    </div>
  );
}