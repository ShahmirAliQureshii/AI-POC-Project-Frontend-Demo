"use client";
import React, { useState } from "react";

type Status = { type: "success" | "error"; msg: string } | null;

export default function Footer(): React.ReactElement {
  const [email, setEmail] = useState<string>("");
  const [status, setStatus] = useState<Status>(null);

  function handleSubscribe(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!email || !/^\S+@\S+\.\S+$/.test(email)) {
      setStatus({ type: "error", msg: "Please enter a valid email." });
      return;
    }
    setStatus({ type: "success", msg: "Thanks — you are subscribed (demo)." });
    setEmail("");
    setTimeout(() => setStatus(null), 4000);
  }

  return (
    <footer className="relative overflow-hidden bg-gradient-to-b from-slate-900 via-slate-950 to-black text-slate-300">
      <div className="max-w-7xl mx-auto px-6 py-14">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-full bg-gradient-to-tr from-indigo-600 to-cyan-400 p-3 shadow-lg">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">AI POC Labs</h3>
                <p className="text-sm text-slate-400">
                  Beautiful face recognition UI — demo frontend for your project.
                </p>
              </div>
            </div>

            <div className="text-sm text-slate-400">
              <p>
                Ship polished demos faster. When you're ready I can wire backend endpoints for uploads,
                inference and realtime dashboard updates.
              </p>
            </div>

            <div className="flex gap-3 mt-3">
              <a aria-label="GitHub" href="#" className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-white/6 hover:bg-white/10 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 .5C5.73.5.98 5.25.98 11.52c0 4.66 3.02 8.62 7.21 10.02.53.1.72-.23.72-.51 0-.25-.01-.92-.01-1.8-2.93.64-3.55-1.38-3.55-1.38-.48-1.22-1.17-1.55-1.17-1.55-.96-.66.07-.65.07-.65 1.07.08 1.64 1.09 1.64 1.09.94 1.62 2.46 1.15 3.06.88.09-.69.37-1.15.67-1.41-2.34-.27-4.8-1.17-4.8-5.2 0-1.15.41-2.09 1.09-2.83-.11-.27-.47-1.36.1-2.85 0 0 .89-.29 2.92 1.08a10.1 10.1 0 012.66-.36c.9 0 1.8.12 2.65.36 2.02-1.37 2.9-1.08 2.9-1.08.58 1.49.22 2.58.11 2.85.68.74 1.09 1.68 1.09 2.83 0 4.04-2.47 4.92-4.82 5.18.38.33.72.98.72 1.98 0 1.43-.01 2.58-.01 2.93 0 .28.19.61.73.51 4.18-1.4 7.19-5.36 7.19-10.02C23.02 5.25 18.27.5 12 .5z" />
                </svg>
              </a>
              <a aria-label="LinkedIn" href="#" className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-white/6 hover:bg-white/10 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M4.98 3.5C3.88 3.5 3 4.38 3 5.48s.88 1.98 1.98 1.98c1.1 0 1.98-.88 1.98-1.98S6.08 3.5 4.98 3.5zM3.5 8.98h3v10.52h-3V8.98zM9.5 8.98h2.88v1.44h.04c.4-.76 1.38-1.56 2.85-1.56 3.05 0 3.61 2.01 3.61 4.62v5.02h-3V14.6c0-1.31-.02-3-1.83-3-1.84 0-2.12 1.43-2.12 2.9v5.01h-3V8.98z" />
                </svg>
              </a>
              <a aria-label="Twitter" href="#" className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-white/6 hover:bg-white/10 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M22 5.92c-.64.28-1.32.47-2.04.56.73-.44 1.28-1.15 1.55-1.99-.68.4-1.43.68-2.23.84A3.49 3.49 0 0015.5 4c-1.93 0-3.5 1.57-3.5 3.5 0 .27.03.54.09.8C8.7 8.14 6.1 6.5 4.3 4.05c-.3.52-.47 1.11-.47 1.74 0 1.2.61 2.26 1.53 2.88-.57-.02-1.11-.18-1.58-.44v.05c0 1.69 1.2 3.1 2.79 3.42-.29.08-.6.12-.92.12-.22 0-.44-.02-.65-.06.44 1.38 1.72 2.39 3.23 2.42A7.02 7.02 0 013 19.54c.57.33 1.23.52 1.93.52 3.3 0 5.1-2.73 5.1-5.1v-.23c.7-.5 1.3-1.13 1.78-1.85-.64.28-1.32.47-2.04.56.73-.44 1.28-1.15 1.55-1.99z" />
                </svg>
              </a>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-white">Product</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><a href="#" className="hover:text-white transition">Overview</a></li>
              <li><a href="#" className="hover:text-white transition">Features</a></li>
              <li><a href="#" className="hover:text-white transition">Pricing</a></li>
              <li><a href="#" className="hover:text-white transition">Docs</a></li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-white">Resources</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><a href="#" className="hover:text-white transition">Blog</a></li>
              <li><a href="#" className="hover:text-white transition">Tutorials</a></li>
              <li><a href="#" className="hover:text-white transition">Support</a></li>
              <li><a href="#" className="hover:text-white transition">Contact Sales</a></li>
            </ul>
          </div>
        </div>

        <div className="mt-10 md:mt-12 bg-gradient-to-r from-white/3 to-white/2 rounded-2xl p-5 md:p-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <h5 className="text-white font-semibold">Stay in the loop</h5>
            <p className="text-sm text-slate-300">Get occasional updates and design previews — frontend demo only.</p>
          </div>

          <form onSubmit={handleSubscribe} className="w-full md:w-auto flex items-center gap-3">
            <label htmlFor="newsletter" className="sr-only">Email</label>
            <input
              id="newsletter"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.currentTarget.value)}
              placeholder="you@company.com"
              className="min-w-0 px-4 py-2 rounded-lg bg-transparent border border-white/10 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold">Subscribe</button>
          </form>

          <div className="w-full md:w-auto text-sm text-slate-400 mt-3 md:mt-0">
            {status ? (
              <span className={status.type === "success" ? "text-emerald-400" : "text-rose-400"}>{status.msg}</span>
            ) : (
              <span>We respect your privacy — no spam.</span>
            )}
          </div>
        </div>

        <div className="mt-8 border-t border-white/6 pt-6 flex flex-col md:flex-row items-center justify-between gap-3 text-sm text-slate-400">
          <div>© {new Date().getFullYear()} <span className="text-white font-semibold">AI POC Labs</span>. All rights reserved.</div>
          <div className="flex items-center gap-4">
            <a href="#" className="hover:text-white transition">Terms</a>
            <a href="#" className="hover:text-white transition">Privacy</a>
            <a href="#" className="hover:text-white transition">Contact</a>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute -right-28 -top-20 w-[480px] h-[480px] bg-indigo-600/10 blur-3xl rounded-full" />
      <div className="pointer-events-none absolute -left-28 -bottom-20 w-[420px] h-[420px] bg-cyan-500/8 blur-3xl rounded-full" />
    </footer>
  );
}