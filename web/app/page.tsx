"use client";

import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

// Backend caps /predict/batch at max_batch_size (32). Chunk below it so a
// dropped folder of any size streams through as several requests.
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
type Item = {
  id: string;
  file: File;
  url: string;
  status: Status;
  prediction?: Prediction;
  error?: string;
};

// "large.cell.carcinoma" -> "Large Cell Carcinoma"
function pretty(label: string): string {
  return label.split(".").join(" ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Recursively collect File objects from a drop that may contain folders.
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
      // readEntries yields in pages; loop until it returns an empty page.
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
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [dragging, setDragging] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/api/v1/health`)
      .then((r) => r.json())
      .then((d) => setOnline(Boolean(d.model_loaded)))
      .catch(() => setOnline(false));
  }, []);

  // `webkitdirectory` isn't in React's input typings — set it on the DOM node.
  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  // Revoke object URLs on unmount.
  const itemsRef = useRef(items);
  itemsRef.current = items;
  useEffect(() => {
    return () => itemsRef.current.forEach((it) => URL.revokeObjectURL(it.url));
  }, []);

  function addFiles(list: FileList | File[] | null) {
    const imgs = (list ? Array.from(list) : []).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (imgs.length === 0) return;
    setItems((prev) => [
      ...prev,
      ...imgs.map((f) => ({
        id: crypto.randomUUID(),
        file: f,
        url: URL.createObjectURL(f),
        status: "pending" as Status,
      })),
    ]);
  }

  function removeItem(id: string) {
    setItems((prev) => {
      const gone = prev.find((it) => it.id === id);
      if (gone) URL.revokeObjectURL(gone.url);
      return prev.filter((it) => it.id !== id);
    });
  }

  function clearAll() {
    items.forEach((it) => URL.revokeObjectURL(it.url));
    setItems([]);
    setOpenId(null);
  }

  async function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    addFiles(await filesFromDrop(e.dataTransfer));
  }

  async function analyze() {
    if (items.length === 0 || loading) return;
    setLoading(true);
    const snapshot = items;
    setItems((prev) =>
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
        setItems((prev) =>
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
        setItems((prev) =>
          prev.map((it) =>
            chunk.some((c) => c.id === it.id) ? { ...it, status: "error", error: msg } : it,
          ),
        );
      }
    }
    setLoading(false);
  }

  const done = items.filter((i) => i.status === "done");
  const failed = items.filter((i) => i.status === "error");
  const dist = done.reduce<Record<string, number>>((acc, it) => {
    const k = it.prediction!.label;
    acc[k] = (acc[k] ?? 0) + 1;
    return acc;
  }, {});
  const open = items.find((i) => i.id === openId) ?? null;
  const openSorted =
    open?.prediction
      ? [...open.prediction.probabilities].sort((a, b) => b.probability - a.probability)
      : [];

  return (
    <main className="page wide">
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
        <input
          ref={fileInput}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          multiple
          hidden
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          hidden
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />

        <div
          className={`drop ${dragging ? "drag" : ""} ${items.length ? "slim" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <div className="drop-hint">
            {!items.length && <span className="plus" aria-hidden>+</span>}
            <p>Drop CT scans or a folder here</p>
            <div className="pick">
              <button className="link" onClick={() => fileInput.current?.click()}>
                Browse files
              </button>
              <span aria-hidden>·</span>
              <button className="link" onClick={() => folderInput.current?.click()}>
                Browse folder
              </button>
            </div>
            {!items.length && <small>PNG · JPG · WEBP · BMP</small>}
          </div>
        </div>

        {items.length > 0 && (
          <>
            <div className="toolbar">
              <div className="summary">
                <strong>{items.length}</strong> scan{items.length > 1 ? "s" : ""}
                {done.length > 0 && <span className="ok-c">{done.length} done</span>}
                {failed.length > 0 && <span className="err-c">{failed.length} failed</span>}
                {Object.entries(dist).map(([label, n]) => (
                  <span key={label} className="chip">
                    {pretty(label)} {n}
                  </span>
                ))}
              </div>
              <div className="actions">
                <button className="btn ghost" onClick={clearAll} disabled={loading}>
                  Clear
                </button>
                <button className="btn" onClick={analyze} disabled={loading || online === false}>
                  {loading ? "Analyzing…" : `Analyze ${items.length}`}
                </button>
              </div>
            </div>

            <ul className="grid">
              {items.map((it) => (
                <li
                  key={it.id}
                  className={`tile ${it.status}`}
                  onClick={() => it.status === "done" && setOpenId(it.id)}
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
                        removeItem(it.id);
                      }}
                    >
                      ×
                    </button>
                    {it.prediction && (
                      <span
                        className="meter"
                        style={{ width: `${it.prediction.confidence * 100}%` }}
                      />
                    )}
                  </div>
                  <div className="tile-body">
                    <span className="tile-name">{it.file.name}</span>
                    {it.status === "done" && it.prediction ? (
                      <span className="tile-label">
                        {pretty(it.prediction.label)}{" "}
                        <em>{(it.prediction.confidence * 100).toFixed(0)}%</em>
                      </span>
                    ) : it.status === "error" ? (
                      <span className="tile-label err-t">{it.error}</span>
                    ) : (
                      <span className="tile-name">
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

      <footer className="footer">
        Research / demo aid only — not for clinical use.
      </footer>

      {open?.prediction && (
        <div className="modal-backdrop" onClick={() => setOpenId(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" aria-label="Close" onClick={() => setOpenId(null)}>
              ×
            </button>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={open.url} alt={open.file.name} className="modal-img" />
            <div className="verdict">
              <span className="verdict-label">{pretty(open.prediction.label)}</span>
              <span className="verdict-conf">
                {(open.prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <ul className="bars">
              {openSorted.map((p) => (
                <li key={p.label} className={p.label === open.prediction!.label ? "top" : ""}>
                  <span className="bar-name">{pretty(p.label)}</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${p.probability * 100}%` }} />
                  </span>
                  <span className="bar-val">{(p.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
            <p className="meta">
              {open.file.name} · inference {open.prediction.inference_time_ms.toFixed(0)} ms
            </p>
          </div>
        </div>
      )}
    </main>
  );
}
