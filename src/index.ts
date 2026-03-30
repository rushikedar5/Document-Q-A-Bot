import express, { Request, Response } from "express";
import { pipeline } from "@xenova/transformers";
import { ChromaClient } from "chromadb";
import OpenAI from "openai";
import * as dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import fs from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdf = require("./pdf-wrapper.cjs");

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const upload = multer({ dest: "uploads/" }); // temp folder for uploaded files

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

console.log("Loading embedding model...");
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
console.log("Embedding model ready.\n");

const chroma = new ChromaClient();

async function getCollection() {
  return chroma.getOrCreateCollection({
    name: "docs",
    embeddingFunction: null as any,
  });
}

async function embed(text: string): Promise<number[]> {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data) as number[];
}

function chunkText(text: string, chunkSize = 200, overlap = 40): string[] {
  const sentences = text
    .split("\n")
    .map(s => s.trim())
    .filter(s => s.length > 0);

  const chunks: string[] = [];
  let current = "";

  for (const sentence of sentences) {
    if ((current + " " + sentence).length > chunkSize && current.length > 0) {
      chunks.push(current.trim());
      const words = current.split(" ");
      current = words.slice(-overlap / 5).join(" ") + " " + sentence;
    } else {
      current = current ? current + " " + sentence : sentence;
    }
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

async function indexText(text: string, docName: string) {
  await chroma.deleteCollection({ name: "docs" }).catch(() => {});
  const collection = await chroma.createCollection({
    name: "docs",
    embeddingFunction: null as any,
  });

  const chunks = chunkText(text);
  const embeddings = await Promise.all(chunks.map(embed));

  await collection.add({
    ids: chunks.map((_, i) => `${docName}-chunk-${i}`),
    embeddings,
    documents: chunks,
    metadatas: chunks.map(() => ({ docName })),
  });

  return chunks.length;
}

// ── POST /ingest — paste raw text ─────────────────────────────────
app.post("/ingest", async (req: Request, res: Response) => {
  const { text, docName }: { text: string; docName: string } = req.body;

  if (!text || !docName) {
    res.status(400).json({ error: "text and docName are required" });
    return;
  }

  try {
    const chunks = await indexText(text, docName);
    res.json({ success: true, chunks, docName });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Ingestion failed" });
  }
});

// ── POST /upload — upload a .txt or .pdf file ─────────────────────
app.post("/upload", upload.single("file"), async (req: Request, res: Response) => {
  const file = req.file;

  if (!file) {
    res.status(400).json({ error: "No file uploaded" });
    return;
  }

  try {
    let text = "";
    const ext = path.extname(file.originalname).toLowerCase();

    if (ext === ".pdf") {
      const buffer = fs.readFileSync(file.path);
      const pdf = (await import("pdf-parse")).default as typeof PdfParse;
      const parsed = await pdf(buffer);
      text = parsed.text;
    } else if (ext === ".txt") {
      text = fs.readFileSync(file.path, "utf-8");
    } else {
      res.status(400).json({ error: "Only .txt and .pdf files are supported" });
      return;
    }

    // Clean up temp file
    fs.unlinkSync(file.path);

    const docName = path.basename(file.originalname, ext);
    const chunks = await indexText(text, docName);

    res.json({ success: true, chunks, docName });
  } catch (err) {
    console.error(err);
    // Clean up temp file on error
    if (req.file) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (unlinkErr) {
        // Ignore cleanup errors
      }
    }
    res.status(500).json({ error: "File processing failed" });
  }
});

// ── POST /ask ─────────────────────────────────────────────────────
app.post("/ask", async (req: Request, res: Response) => {
  const { question }: { question: string } = req.body;

  if (!question) {
    res.status(400).json({ error: "question is required" });
    return;
  }

  try {
    const collection = await getCollection();
    const count = await collection.count();

    if (count === 0) {
      res.status(400).json({ error: "No document indexed yet. Please ingest a document first." });
      return;
    }

    const questionVector = await embed(question);
    const results = await collection.query({
      queryEmbeddings: [questionVector],
      nResults: 3,
    });

    const relevantChunks = (results.documents[0] ?? []).filter(Boolean) as string[];
    const context = relevantChunks.join("\n\n");

    const response = await client.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      max_tokens: 512,
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant. Answer the user's question using ONLY the context below. If the answer is not in the context, say "I don't have that information."

Context:
${context}`,
        },
        { role: "user", content: question },
      ],
    });

    const answer = response.choices[0].message.content ?? "";
    res.json({ answer, chunks: relevantChunks });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Query failed" });
  }
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));