"use client";

import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

type Prob = { label: string; probability: number };
type Prediction = {
  label: string;
  confidence: number;
  probabilities: Prob[];
  inference_time_ms: number;
};
type BatchItem = {
  filename: string | null;
  success: boolean;
  prediction: Prediction | null;
  error: string | null;
};

// "large.cell.carcinoma" -> "Large Cell Carcinoma"
function pretty(label: string): string {
  return label.split(".").join(" ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [result, setResult] = useState<Prediction | null>(null); // single upload
  const [batch, setBatch] = useState<BatchItem[] | null>(null); // multi upload
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/api/v1/health`)
      .then((r) => r.json())
      .then((d) => setOnline(Boolean(d.model_loaded)))
      .catch(() => setOnline(false));
  }, []);

  // Build object-URL previews whenever the file set changes; revoke on cleanup.
  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  function choose(list: FileList | null) {
    setResult(null);
    setBatch(null);
    setError(null);
    const picked = list
      ? Array.from(list).filter((f) => f.type.startsWith("image/"))
      : [];
    setFiles(picked);
  }

  async function analyze() {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBatch(null);
    try {
      const headers = API_KEY ? { "X-API-Key": API_KEY } : undefined;
      if (files.length === 1) {
        const form = new FormData();
        form.append("file", files[0]);
        const res = await fetch(`${API}/api/v1/predict`, {
          method: "POST",
          body: form,
          headers,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
        setResult(data as Prediction);
      } else {
        const form = new FormData();
        // Field name must be "files" to match the /predict/batch endpoint.
        files.forEach((f) => form.append("files", f));
        const res = await fetch(`${API}/api/v1/predict/batch`, {
          method: "POST",
          body: form,
          headers,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
        setBatch((data.results ?? []) as BatchItem[]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    choose(e.dataTransfer.files);
  }

  const single = files.length === 1;
  const sorted = result
    ? [...result.probabilities].sort((a, b) => b.probability - a.probability)
    : [];

  return (
    <main className="page">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden>◧</span>
          <h1>PulmoScan&nbsp;AI</h1>
        </div>
        <p className="tagline">Chest CT-scan classifier</p>
        <span className={`status ${online ? "ok" : online === false ? "down" : ""}`}>
          {online === null ? "checking…" : online ? "model online" : "offline"}
        </span>
      </header>

      <section className="card">
        <div
          className={`drop ${dragging ? "drag" : ""} ${single ? "has" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {single ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previews[0]} alt="scan preview" className="preview" />
          ) : files.length > 1 ? (
            <div className="thumbs">
              {previews.map((src, i) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img key={i} src={src} alt={`scan ${i + 1}`} className="thumb" />
              ))}
            </div>
          ) : (
            <div className="drop-hint">
              <span className="plus" aria-hidden>+</span>
              <p>Drop CT scans here, or click to browse</p>
              <small>One or many · PNG · JPG · WEBP · BMP</small>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            multiple
            hidden
            onChange={(e) => choose(e.target.files)}
          />
        </div>

        <div className="actions">
          <button
            className="btn"
            onClick={analyze}
            disabled={files.length === 0 || loading || online === false}
          >
            {loading
              ? "Analyzing…"
              : single
                ? "Analyze"
                : `Analyze ${files.length} scans`}
          </button>
          {files.length > 0 && (
            <button className="btn ghost" onClick={() => choose(null)} disabled={loading}>
              Clear
            </button>
          )}
        </div>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <div className="verdict">
              <span className="verdict-label">{pretty(result.label)}</span>
              <span className="verdict-conf">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <ul className="bars">
              {sorted.map((p) => (
                <li key={p.label} className={p.label === result.label ? "top" : ""}>
                  <span className="bar-name">{pretty(p.label)}</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${p.probability * 100}%` }} />
                  </span>
                  <span className="bar-val">{(p.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
            <p className="meta">inference {result.inference_time_ms.toFixed(0)} ms</p>
          </div>
        )}

        {batch && (
          <ul className="batch">
            {batch.map((item, i) => (
              <li key={i} className={`batch-item ${item.success ? "" : "err"}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previews[i]} alt={item.filename ?? `scan ${i + 1}`} className="batch-thumb" />
                <div className="batch-info">
                  <span className="batch-name">{item.filename ?? `scan ${i + 1}`}</span>
                  <span className="batch-sub">
                    {item.success && item.prediction
                      ? pretty(item.prediction.label)
                      : item.error ?? "Failed"}
                  </span>
                </div>
                {item.success && item.prediction && (
                  <span className="batch-conf">
                    {(item.prediction.confidence * 100).toFixed(1)}%
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <footer className="footer">
        Research / demo aid only — not for clinical use.
      </footer>
    </main>
  );
}
