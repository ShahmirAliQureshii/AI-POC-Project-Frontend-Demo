import { useState } from "react";


export default function Header({active, setActive}) {
  const tabs = ["Home","Training", "Model Viewer", "Image", "Video", "Dashboard"];

  return (
    <header className="flex items-center justify-between gap-6">
      <div className="flex items-center gap-4">
        <div className="rounded-full bg-gradient-to-tr from-indigo-600 to-cyan-400 p-3 shadow-lg">
          <svg
          onClick={()=> setActive("Home")}
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-white cursor-pointer"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.5"
              d="M15 10l4.553-2.276A2 2 0 0122 9.618v4.764a2 2 0 01-2.447 1.894L15 14M9 14l-4.553 2.276A2 2 0 013 15.382V10.618a2 2 0 012.447-1.894L9 10M9 10v4M15 14v-4"
            />
          </svg>
        </div>
        <div>
          <h1 className="text-2xl font-semibold cursor-pointer">Face Recognition Web</h1>
          <p className="text-sm text-slate-500 cursor-pointer">
            Modern UI for the desktop face recognition system — frontend-only
            demo
          </p>
        </div>
      </div>

      <nav className="hidden sm:flex items-center gap-2 rounded-xl bg-white/60 px-2 py-1 shadow-sm backdrop-blur">
        {tabs.map((t) => (
          <button
            key={t}
            onClick={() => setActive(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              active === t
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-700 hover:bg-slate-100"
            }`}
          >
            {t}
          </button>
        ))}
      </nav>
    </header>
  );
}
