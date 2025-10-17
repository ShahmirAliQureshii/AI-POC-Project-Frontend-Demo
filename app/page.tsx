"use client";

import React, { useState, useMemo } from "react";
import Header from "./components/header";
import HeroSection from "./components/hero";
import Footer from "./components/footer";
import MainPanel from "./components/MainPanel";
import RightAside from "./components/RightAside";
import { processImagesOnServer } from "../lib/api";

export default function Home(): React.ReactElement {
  const [active, setActive] = useState<string>("Dashboard");
  const [message, setMessage] = useState<string | null>(null);

  const sampleStats = useMemo(
    () => ({
      detections: 128,
      emotions: { happy: 56, neutral: 40, sad: 12 },
      cards: { red: 3, yellow: 5, green: 8 },
    }),
    []
  );

  const [recognized, setRecognized] = useState<Array<any>>([]);
  const [unrecognized, setUnrecognized] = useState<Array<any>>([]);

  function handleFilePlaceholder(e: React.ChangeEvent<HTMLInputElement> | null, label: string) {
    const count = e?.target?.files?.length || 0;
    setMessage(`${label}: ${count} file(s) selected (placeholder)`);
    setTimeout(() => setMessage(null), 2500);
  }

  function processImages(files?: FileList | null) {
    const f = Array.from(files || []);
    if (!f.length) {
      setMessage("No files selected");
      setTimeout(() => setMessage(null), 2000);
      return;
    }

    // call backend
    processImagesOnServer(f)
      .then((res) => {
        if (res?.ok && res.results) {
          setRecognized(res.results.recognized ?? []);
          setUnrecognized(res.results.unrecognized ?? []);
          setMessage(`Processed ${f.length} images (server)`);
        } else {
          setMessage("Processing failed (server)");
        }
      })
      .catch((err) => {
        console.error("processImages error", err);
        setMessage("Server error processing images");
      })
      .finally(() => setTimeout(() => setMessage(null), 2500));
  }

  function processVideoDetections(demoList: Array<any> = []) {
    const rec = demoList.filter((d) => d.status === "recognized");
    const unrec = demoList.filter((d) => d.status === "unrecognized");
    setRecognized(rec);
    setUnrecognized(unrec);
    setMessage(`Video detections updated (demo)`);
    setTimeout(() => setMessage(null), 2000);
  }

  const showAside = active === "Home";

  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      <Header active={active} setActive={setActive} />
      {active === "Home" && <HeroSection active={active} setActive={setActive} />}

      <section
        className={`mt-8 grid grid-cols-1 ${showAside ? "lg:grid-cols-4" : "lg:grid-cols-1"} gap-6`}
      >
        <div className={showAside ? "lg:col-span-3 space-y-6" : "space-y-6"}>
          <MainPanel
            active={active}
            recognized={recognized}
            unrecognized={unrecognized}
            sampleStats={sampleStats}
            handleFilePlaceholder={handleFilePlaceholder}
            processImages={processImages}
            processVideoDetections={processVideoDetections}
          />
        </div>

        {showAside && <RightAside setActive={setActive} message={message} />}
      </section>

      <Footer />
    </main>
  );
}