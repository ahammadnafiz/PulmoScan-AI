"use client";

import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

const CHUNK = 16;

type Prob = { label: string; probability: number };
type Prediction = {
  label: string;
  confidence: number;
  probabilities: Prob[];
  inference_time_ms: number;
};
type ApiBatchItem = {
  filename: string | null;
  success: boolean;
  prediction: Prediction | null;
  error: string | null;
};

type Status = "pending" | "analyzing" | "done" | "error";

// Single scan state item
type SingleItem = {
  file: File;
  url: string;
  status: Status;
  prediction?: Prediction;
  error?: string;
};

// Batch scan state item
type BatchItem = {
  id: string;
  file: File;
  url: string;
  status: Status;
  prediction?: Prediction;
  error?: string;
};

function pretty(label: string): string {
  return label.split(".").join(" ").replace(/\b\w/g, (c) => c.toUpperCase());
}

async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries = Array.from(dt.items)
    .map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null))
    .filter((e): e is FileSystemEntry => e !== null);

  if (entries.length === 0) return Array.from(dt.files);

  const out: File[] = [];
  async function walk(entry: FileSystemEntry): Promise<void> {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) =>
        (entry as FileSystemFileEntry).file(res, rej),
      );
      out.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      let page: FileSystemEntry[];
      do {
        page = await new Promise<FileSystemEntry[]>((res, rej) =>
          reader.readEntries(res, rej),
        );
        for (const e of page) await walk(e);
      } while (page.length > 0);
    }
  }
  await Promise.all(entries.map(walk));
  return out;
}

export default function Home() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [online, setOnline] = useState<boolean | null>(null);

  // --- Single Mode State ---
  const [singleItem, setSingleItem] = useState<SingleItem | null>(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleDragging, setSingleDragging] = useState(false);
  const singleFileInput = useRef<HTMLInputElement>(null);

  // --- Batch Mode State ---
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchDragging, setBatchDragging] = useState(false);
  const [openBatchId, setOpenBatchId] = useState<string | null>(null);
  
  const batchFileInput = useRef<HTMLInputElement>(null);
  const batchFolderInput = useRef<HTMLInputElement>(null);

  // Check health and if model is loaded on boot
  useEffect(() => {
    fetch(`${API}/api/v1/health`)
      .then((r) => r.json())
      .then((d) => setOnline(Boolean(d.model_loaded)))
      .catch(() => setOnline(false));
  }, []);

  // Set webkitdirectory on folder input DOM node
  useEffect(() => {
    batchFolderInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  // Revoke Object URLs on unmount
  useEffect(() => {
    return () => {
      if (singleItem) URL.revokeObjectURL(singleItem.url);
      batchItems.forEach((it) => URL.revokeObjectURL(it.url));
    };
  }, [singleItem, batchItems]);

  // --- Single Mode Functions ---
  function handleSingleSelect(file: File) {
    if (!file.type.startsWith("image/")) return;
    if (singleItem) URL.revokeObjectURL(singleItem.url);
    setSingleItem({
      file,
      url: URL.createObjectURL(file),
      status: "pending",
    });
  }

  async function analyzeSingle() {
    if (!singleItem || singleLoading) return;
    setSingleLoading(true);
    setSingleItem((prev) => prev ? { ...prev, status: "analyzing", prediction: undefined, error: undefined } : null);

    const form = new FormData();
    form.append("file", singleItem.file);

    const headers = API_KEY ? { "X-API-Key": API_KEY } : undefined;

    try {
      const res = await fetch(`${API}/api/v1/predict`, {
        method: "POST",
        body: form,
        headers,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
      
      setSingleItem((prev) => prev ? { ...prev, status: "done", prediction: data, error: undefined } : null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Analysis failed";
      setSingleItem((prev) => prev ? { ...prev, status: "error", error: msg } : null);
    } finally {
      setSingleLoading(false);
    }
  }

  function clearSingle() {
    if (singleItem) URL.revokeObjectURL(singleItem.url);
    setSingleItem(null);
  }

  // --- Batch Mode Functions ---
  function addBatchFiles(list: FileList | File[] | null) {
    const imgs = (list ? Array.from(list) : []).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (imgs.length === 0) return;
    setBatchItems((prev) => [
      ...prev,
      ...imgs.map((f) => ({
        id: crypto.randomUUID(),
        file: f,
        url: URL.createObjectURL(f),
        status: "pending" as Status,
      })),
    ]);
  }

  function removeBatchItem(id: string) {
    setBatchItems((prev) => {
      const gone = prev.find((it) => it.id === id);
      if (gone) URL.revokeObjectURL(gone.url);
      return prev.filter((it) => it.id !== id);
    });
    if (openBatchId === id) setOpenBatchId(null);
  }

  function clearAllBatch() {
    batchItems.forEach((it) => URL.revokeObjectURL(it.url));
    setBatchItems([]);
    setOpenBatchId(null);
  }

  async function analyzeBatch() {
    if (batchItems.length === 0 || batchLoading) return;
    setBatchLoading(true);
    const snapshot = batchItems;
    setBatchItems((prev) =>
      prev.map((it) => ({ ...it, status: "analyzing", prediction: undefined, error: undefined })),
    );
    const headers = API_KEY ? { "X-API-Key": API_KEY } : undefined;

    for (let start = 0; start < snapshot.length; start += CHUNK) {
      const chunk = snapshot.slice(start, start + CHUNK);
      const form = new FormData();
      chunk.forEach((it) => form.append("files", it.file));

      try {
        const res = await fetch(`${API}/api/v1/predict/batch`, {
          method: "POST",
          body: form,
          headers,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
        const results = (data.results ?? []) as ApiBatchItem[];

        setBatchItems((prev) =>
          prev.map((it) => {
            const idx = chunk.findIndex((c) => c.id === it.id);
            if (idx === -1) return it;
            const r = results[idx];
            if (r?.success && r.prediction)
              return { ...it, status: "done", prediction: r.prediction, error: undefined };
            return { ...it, status: "error", error: r?.error ?? "No result" };
          }),
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Request failed";
        setBatchItems((prev) =>
          prev.map((it) =>
            chunk.some((c) => c.id === it.id) ? { ...it, status: "error", error: msg } : it,
          ),
        );
      }
    }
    setBatchLoading(false);
  }

  // Derived Batch variables
  const batchDone = batchItems.filter((i) => i.status === "done");
  const batchFailed = batchItems.filter((i) => i.status === "error");
  const batchDist = batchDone.reduce<Record<string, number>>((acc, it) => {
    const k = it.prediction!.label;
    acc[k] = (acc[k] ?? 0) + 1;
    return acc;
  }, {});

  const openBatchItem = batchItems.find((i) => i.id === openBatchId) ?? null;
  const openBatchSorted = openBatchItem?.prediction
    ? [...openBatchItem.prediction.probabilities].sort((a, b) => b.probability - a.probability)
    : [];

  const singleSorted = singleItem?.prediction
    ? [...singleItem.prediction.probabilities].sort((a, b) => b.probability - a.probability)
    : [];

  return (
    <main className="page wide">
      <header className="header" style={{ marginBottom: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", width: "100%" }}>
          <div>
            <div className="brand">
              <span className="logo" aria-hidden style={{ color: "#319795", fontWeight: "bold" }}>◧</span>
              <h1>PulmoScan&nbsp;AI</h1>
            </div>
            <p className="tagline">Production chest CT-scan classifier & observability platform</p>
          </div>
          <span className={`status ${online ? "ok" : online === false ? "down" : ""}`} style={{ margin: 0 }}>
            {online === null ? "checking…" : online ? "model online" : "offline"}
          </span>
        </div>

        {/* --- Custom Tab Switcher --- */}
        <div style={{
          display: "flex",
          gap: "8px",
          marginTop: "20px",
          background: "#f0f0f0",
          padding: "4px",
          borderRadius: "10px",
          alignSelf: "flex-start"
        }}>
          <button
            onClick={() => setMode("single")}
            style={{
              font: "inherit",
              fontSize: "13px",
              fontWeight: "600",
              padding: "6px 16px",
              borderRadius: "8px",
              border: "none",
              cursor: "pointer",
              background: mode === "single" ? "#ffffff" : "transparent",
              color: mode === "single" ? "#0a0a0a" : "#737373",
              boxShadow: mode === "single" ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
              transition: "all 0.15s ease"
            }}
          >
            🔎 Single Scan Mode
          </button>
          <button
            onClick={() => setMode("batch")}
            style={{
              font: "inherit",
              fontSize: "13px",
              fontWeight: "600",
              padding: "6px 16px",
              borderRadius: "8px",
              border: "none",
              cursor: "pointer",
              background: mode === "batch" ? "#ffffff" : "transparent",
              color: mode === "batch" ? "#0a0a0a" : "#737373",
              boxShadow: mode === "batch" ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
              transition: "all 0.15s ease"
            }}
          >
            🗂️ Batch & Folder Mode
          </button>
        </div>
      </header>

      {/* ======================================================== */}
      {/* 🔎 SINGLE SCAN MODE                                      */}
      {/* ======================================================== */}
      {mode === "single" && (
        <section className="card">
          <input
            ref={singleFileInput}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            hidden
            onChange={(e) => {
              if (e.target.files?.[0]) handleSingleSelect(e.target.files[0]);
              e.target.value = "";
            }}
          />

          {!singleItem ? (
            <div
              className={`drop ${singleDragging ? "drag" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setSingleDragging(true);
              }}
              onDragLeave={() => setSingleDragging(false)}
              onDrop={async (e) => {
                e.preventDefault();
                setSingleDragging(false);
                const files = await filesFromDrop(e.dataTransfer);
                if (files?.[0]) handleSingleSelect(files[0]);
              }}
              onClick={() => singleFileInput.current?.click()}
            >
              <div className="drop-hint">
                <span className="plus" aria-hidden>+</span>
                <p>Drag & drop a single chest CT scan here</p>
                <div className="pick">
                  <button className="link" onClick={(e) => { e.stopPropagation(); singleFileInput.current?.click(); }}>
                    Browse computer
                  </button>
                </div>
                <small>PNG · JPG · WEBP · BMP</small>
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#000", maxHeight: "360px" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={singleItem.url}
                  alt={singleItem.file.name}
                  style={{ width: "100%", height: "100%", maxHeight: "360px", objectFit: "contain", display: "block" }}
                />
                
                {singleItem.status === "analyzing" && (
                  <div style={{
                    position: "absolute",
                    inset: 0,
                    background: "rgba(0,0,0,0.5)",
                    display: "grid",
                    placeItems: "center"
                  }}>
                    <span className="spinner" style={{ width: "40px", height: "40px" }} />
                  </div>
                )}

                <button
                  onClick={clearSingle}
                  disabled={singleLoading}
                  style={{
                    position: "absolute",
                    top: "12px",
                    right: "12px",
                    width: "28px",
                    height: "28px",
                    borderRadius: "999px",
                    background: "rgba(0,0,0,0.7)",
                    color: "#fff",
                    border: "none",
                    cursor: "pointer",
                    display: "grid",
                    placeItems: "center",
                    fontSize: "18px",
                    fontWeight: "bold"
                  }}
                  aria-label="Remove"
                >
                  ×
                </button>
              </div>

              {/* Bottom Actions Bar */}
              {singleItem.status === "pending" && (
                <div className="actions">
                  <button className="btn ghost" onClick={clearSingle} disabled={singleLoading}>
                    Clear
                  </button>
                  <button className="btn" onClick={analyzeSingle} disabled={singleLoading || online === false}>
                    {singleLoading ? "Analyzing…" : "Analyze CT Scan"}
                  </button>
                </div>
              )}

              {/* Error Box */}
              {singleItem.status === "error" && (
                <div className="error" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span>⚠️ {singleItem.error}</span>
                  <button className="btn" style={{ flex: "0 0 auto", padding: "6px 12px", fontSize: "12px" }} onClick={analyzeSingle}>
                    Retry
                  </button>
                </div>
              )}

              {/* Prediction result view */}
              {singleItem.status === "done" && singleItem.prediction && (
                <div className="result" style={{ animation: "fadeIn 0.25s ease-out" }}>
                  <div className="verdict" style={{ border: "none", padding: 0 }}>
                    <div style={{ display: "flex", flexDirection: "column" }}>
                      <span style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.04em", color: "#737373" }}>
                        Diagnosis Verdict
                      </span>
                      <span className="verdict-label" style={{ fontSize: "24px", color: "#000" }}>
                        {pretty(singleItem.prediction.label)}
                      </span>
                    </div>
                    <span className="verdict-conf" style={{ fontSize: "32px", fontWeight: "700" }}>
                      {(singleItem.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Custom horizontal progress bars */}
                  <ul className="bars" style={{ marginTop: "10px" }}>
                    {singleSorted.map((p) => (
                      <li key={p.label} className={p.label === singleItem.prediction!.label ? "top" : ""}>
                        <span className="bar-name" style={{ fontSize: "13px" }}>{pretty(p.label)}</span>
                        <span className="bar-track" style={{ height: "8px" }}>
                          <span
                            className="bar-fill"
                            style={{
                              width: `${p.probability * 100}%`,
                              background: p.label === singleItem.prediction!.label ? "#319795" : "#737373"
                            }}
                          />
                        </span>
                        <span className="bar-val" style={{ fontSize: "13px" }}>{(p.probability * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>

                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "12px",
                    color: "#737373",
                    borderTop: "1px solid var(--line)",
                    paddingTop: "12px",
                    marginTop: "6px"
                  }}>
                    <span>📄 {singleItem.file.name}</span>
                    <span>⚡ inference: <strong>{singleItem.prediction.inference_time_ms.toFixed(0)} ms</strong></span>
                  </div>

                  <button className="btn ghost" onClick={clearSingle} style={{ alignSelf: "center", marginTop: "10px", width: "120px" }}>
                    Scan New
                  </button>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* ======================================================== */}
      {/* 🗂️ BATCH & FOLDER MODE                                   */}
      {/* ======================================================== */}
      {mode === "batch" && (
        <section className="card">
          <input
            ref={batchFileInput}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            multiple
            hidden
            onChange={(e) => {
              addBatchFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={batchFolderInput}
            type="file"
            hidden
            onChange={(e) => {
              addBatchFiles(e.target.files);
              e.target.value = "";
            }}
          />

          <div
            className={`drop ${batchDragging ? "drag" : ""} ${batchItems.length ? "slim" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setBatchDragging(true);
            }}
            onDragLeave={() => setBatchDragging(false)}
            onDrop={async (e) => {
              e.preventDefault();
              setBatchDragging(false);
              addBatchFiles(await filesFromDrop(e.dataTransfer));
            }}
          >
            <div className="drop-hint">
              {!batchItems.length && <span className="plus" aria-hidden>+</span>}
              <p>Drop multiple CT scans or an entire folder here</p>
              <div className="pick">
                <button className="link" onClick={(e) => { e.stopPropagation(); batchFileInput.current?.click(); }}>
                  Browse files
                </button>
                <span aria-hidden>·</span>
                <button className="link" onClick={(e) => { e.stopPropagation(); batchFolderInput.current?.click(); }}>
                  Browse folders
                </button>
              </div>
              {!batchItems.length && <small>PNG · JPG · WEBP · BMP</small>}
            </div>
          </div>

          {batchItems.length > 0 && (
            <>
              <div className="toolbar" style={{ borderTop: "1px solid var(--line)", paddingTop: "16px" }}>
                <div className="summary" style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  <strong>{batchItems.length}</strong> scan{batchItems.length > 1 ? "s" : ""}
                  {batchDone.length > 0 && <span className="ok-c">{batchDone.length} done</span>}
                  {batchFailed.length > 0 && <span className="err-c">{batchFailed.length} failed</span>}
                  
                  {Object.entries(batchDist).map(([label, n]) => (
                    <span key={label} className="chip" style={{ background: "#e6fffa", color: "#319795", border: "1px solid #b2f5ea" }}>
                      {pretty(label)}: <strong>{n}</strong>
                    </span>
                  ))}
                </div>
                
                <div className="actions">
                  <button className="btn ghost" onClick={clearAllBatch} disabled={batchLoading}>
                    Clear
                  </button>
                  <button className="btn" onClick={analyzeBatch} disabled={batchLoading || online === false}>
                    {batchLoading ? "Analyzing…" : `Analyze Batch (${batchItems.length})`}
                  </button>
                </div>
              </div>

              <ul className="grid">
                {batchItems.map((it) => (
                  <li
                    key={it.id}
                    className={`tile ${it.status}`}
                    onClick={() => it.status === "done" && setOpenBatchId(it.id)}
                    style={{ transition: "transform 0.15s, box-shadow 0.15s" }}
                  >
                    <div className="tile-media">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={it.url} alt={it.file.name} className="tile-img" />
                      
                      <span className={`badge ${it.status}`} aria-label={it.status}>
                        {it.status === "analyzing" ? (
                          <span className="spinner" />
                        ) : it.status === "done" ? (
                          "✓"
                        ) : it.status === "error" ? (
                          "!"
                        ) : (
                          ""
                        )}
                      </span>
                      
                      <button
                        className="remove"
                        aria-label="Remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeBatchItem(it.id);
                        }}
                      >
                        ×
                      </button>

                      {it.prediction && (
                        <span
                          className="meter"
                          style={{
                            width: `${it.prediction.confidence * 100}%`,
                            background: "#319795"
                          }}
                        />
                      )}
                    </div>
                    
                    <div className="tile-body">
                      <span className="tile-name">{it.file.name}</span>
                      {it.status === "done" && it.prediction ? (
                        <span className="tile-label" style={{ color: "#319795", fontWeight: "600" }}>
                          {pretty(it.prediction.label)}{" "}
                          <em>{(it.prediction.confidence * 100).toFixed(0)}%</em>
                        </span>
                      ) : it.status === "error" ? (
                        <span className="tile-label err-t">{it.error}</span>
                      ) : (
                        <span className="tile-name" style={{ color: "#737373" }}>
                          {it.status === "analyzing" ? "analyzing…" : "ready"}
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}

      <footer className="footer">
        PulmoScan Observability Suite · Research / demo aid only — not for clinical use.
      </footer>

      {/* ======================================================== */}
      {/* 🎂 BATCH RESULT DETAIL MODAL                              */}
      {/* ======================================================== */}
      {openBatchItem?.prediction && (
        <div className="modal-backdrop" onClick={() => setOpenBatchId(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ animation: "scaleUp 0.2s cubic-bezier(0.16, 1, 0.3, 1)" }}>
            <button className="modal-close" aria-label="Close" onClick={() => setOpenBatchId(null)}>
              ×
            </button>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={openBatchItem.url} alt={openBatchItem.file.name} className="modal-img" />
            <div className="verdict">
              <span className="verdict-label">{pretty(openBatchItem.prediction.label)}</span>
              <span className="verdict-conf" style={{ color: "#319795" }}>
                {(openBatchItem.prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            
            <ul className="bars">
              {openBatchSorted.map((p) => (
                <li key={p.label} className={p.label === openBatchItem.prediction!.label ? "top" : ""}>
                  <span className="bar-name">{pretty(p.label)}</span>
                  <span className="bar-track">
                    <span
                      className="bar-fill"
                      style={{
                        width: `${p.probability * 100}%`,
                        background: p.label === openBatchItem.prediction!.label ? "#319795" : "#737373"
                      }}
                    />
                  </span>
                  <span className="bar-val">{(p.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
            
            <p className="meta">
              {openBatchItem.file.name} · inference {openBatchItem.prediction.inference_time_ms.toFixed(0)} ms
            </p>
          </div>
        </div>
      )}
    </main>
  );
}
