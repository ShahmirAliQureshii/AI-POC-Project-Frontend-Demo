"use client";

import { useState, useMemo } from "react";
import Header from "./components/header";
import HeroSection from "./components/hero";
import Footer from "./components/footer";
import ResultsTabs from "./ResultsTabs";
import VideoPlayer from "./VideoPlayer";
import MainPanel from "./components/MainPanel";
import RightAside from "./components/RightAside";

export default function Home() {
  const [active, setActive] = useState("Dashboard");
  const [message, setMessage] = useState(null);

  const sampleStats = useMemo(
    () => ({
      detections: 128,
      emotions: { happy: 56, neutral: 40, sad: 12 },
      cards: { red: 3, yellow: 5, green: 8 },
    }),
    []
  );

  const [recognized, setRecognized] = useState([]);
  const [unrecognized, setUnrecognized] = useState([]);

  function handleFilePlaceholder(e, label) {
    const count = e?.target?.files?.length || 0;
    setMessage(`${label}: ${count} file(s) selected (placeholder)`);
    setTimeout(() => setMessage(null), 2500);
  }

  function processImages(files) {
    const f = Array.from(files || []);
    const rec = f
      .filter((_, i) => i % 2 === 0)
      .map((file, i) => ({
        id: `img-rec-${i}`,
        name: file.name || `image-${i + 1}`,
        thumb: URL.createObjectURL(file),
      }));
    const unrec = f
      .filter((_, i) => i % 2 !== 0)
      .map((file, i) => ({
        id: `img-un-${i}`,
        name: file.name || `image-${i + 1}`,
        thumb: URL.createObjectURL(file),
      }));
    setRecognized(rec);
    setUnrecognized(unrec);
    setMessage(`Processed ${f.length} image(s) (demo)`);
    setTimeout(() => setMessage(null), 2500);
  }

  function processVideoDetections(demoList = []) {
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
      {active === "Home" && (
        <HeroSection active={active} setActive={setActive} />
      )}

      <section
        className={`mt-8 grid grid-cols-1 ${
          showAside ? "lg:grid-cols-4" : "lg:grid-cols-1"
        } gap-6`}
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
