"use client";

export default function RightAside({ setActive, message }) {
  return (
    <aside className="space-y-6">
      <div className="rounded-xl bg-white p-4 shadow">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-500">Quick Actions</div>
            <div className="font-medium">Shortcuts</div>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-2">
          <button
            onClick={() => setActive("Training")}
            className="w-full text-left px-3 py-2 rounded-lg bg-indigo-50 hover:bg-indigo-600 hover:text-white"
          >
            Open Training
          </button>
          <button
            onClick={() => setActive("Image")}
            className="w-full text-left px-3 py-2 rounded-lg bg-indigo-50 hover:bg-indigo-600 hover:text-white"
          >
            Process Images
          </button>
          <button
            onClick={() => setActive("Video")}
            className="w-full text-left px-3 py-2 rounded-lg bg-indigo-50 hover:bg-indigo-600 hover:text-white"
          >
            Process Video
          </button>
        </div>
      </div>

      <div className="rounded-xl bg-white p-4 shadow">
        <div className="text-sm text-slate-500">Project Info</div>
        <div className="mt-2 text-xs text-slate-600">
          Demo frontend for the desktop Face Recognition application. No backend
          yet.
        </div>
        <div className="mt-3">
          <button className="w-full px-3 py-2 rounded-lg bg-slate-100">
            Export UI Mock
          </button>
        </div>
      </div>

      <div className="rounded-xl bg-white p-4 shadow">
        <div className="text-sm text-slate-500">Messages</div>
        <div className="mt-2 text-xs text-slate-600">
          {message ?? "No recent messages"}
        </div>
      </div>
    </aside>
  );
}
